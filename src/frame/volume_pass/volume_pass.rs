use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, SquareMatrix, Deg, Angle, Point3, Matrix};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImmutableImage, MipmapsCount, ImageDimensions};
use vulkano::impl_vertex;
use vulkano::pipeline::graphics::color_blend::{ColorBlendState, AttachmentBlend, BlendOp, BlendFactor};
use vulkano::sampler::{SamplerCreateInfo, Sampler, Filter, SamplerAddressMode};
use std::{ sync::Arc};

use vulkano::render_pass::RenderPass;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        SecondaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Queue,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass,
};

use crate::scene_pkg::volume::Volume;

#[repr(C)]
#[derive(Clone)]
struct Buffers {
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

pub struct VolumePass {
    gfx_queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    buffers: Buffers,
    volume_set: Arc<PersistentDescriptorSet>,
    volume: Volume
}

impl VolumePass {
    /// Initializes a triangle drawing system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        volume: Volume,
    ) -> VolumePass {
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let buffers = VolumePass::create_buffers(gfx_queue.clone(), volume.clone());

        let pipeline = VolumePass::create_pipeline(gfx_queue.clone(), subpass.clone());

        let volume_set = VolumePass::create_volume_set(pipeline.clone(), gfx_queue.clone(), volume.clone());

        VolumePass {
            gfx_queue,
            pipeline,
            subpass,
            buffers,
            volume,
            volume_set
        }
    }

    pub fn draw(
        &mut self,
        viewport_dimensions: [u32; 2],
        world: Matrix4<f32>,
        view: Matrix4<f32>,
        eye_pos: Point3<f32>

    ) -> SecondaryAutoCommandBuffer {

        let focal_length = 1.0 / Deg(72.0 / 2.0).tan();
        let push_constants_fs;
        {
            let ray_origin = world.transpose() *  eye_pos.to_homogeneous();
            push_constants_fs = fs::ty::PushConstants {
                light_position: [0.25, 1.0, 3.0],
                light_intensity: [15.0, 15.0, 15.0],
                absorption: 1.0,
                model_view: (view).into(),
                focal_length: focal_length,
                window_size: [viewport_dimensions[0]  as f32, viewport_dimensions[1] as f32],
                ray_origin: ray_origin.truncate().into(),
                _dummy0: [1, 1, 1, 1],
                _dummy1: [1, 1, 1, 1],
            };
        }

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let mut builder = AutoCommandBufferBuilder::secondary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .set_viewport(0, [viewport])
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.volume_set.clone(),
            )
            .push_constants(self.pipeline.layout().clone(), 0, push_constants_fs)
            
            .bind_vertex_buffers(0, self.buffers.vertex_buffer.clone())
            .draw(self.buffers.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    // private methods

    fn create_volume_set(pipeline: Arc<GraphicsPipeline>, gfx_queue: Arc<Queue>, volume: Volume) -> Arc<PersistentDescriptorSet> {
        
        let dim3 = ImageDimensions::Dim3d { width: volume.dimension[0], height: volume.dimension[1], depth: volume.dimension[2] };
        let (image, future) = ImmutableImage::from_iter(
            volume.pixel_data,
            dim3,
            MipmapsCount::One,
            Format::D16_UNORM,
            gfx_queue.clone(),
        )
        .unwrap();
        let image_view = ImageView::new_default(image).unwrap();
        
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                ..Default::default()
            },
        )
        .unwrap();

        let write_descriptor = WriteDescriptorSet::image_view_sampler(0, image_view, sampler);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let volume_set = PersistentDescriptorSet::new(
            layout.clone(),
            [write_descriptor],
        )
        .unwrap();

        volume_set
    }

    fn create_pipeline(gfx_queue: Arc<Queue>, subpass: Subpass) -> Arc<GraphicsPipeline> {
        let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
        let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend(
                AttachmentBlend {
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                },
            ))
            .render_pass(subpass.clone())
            .build(gfx_queue.device().clone())
            .unwrap()
    }

    fn create_buffers(gfx_queue: Arc<Queue>, _volume: Volume) -> Buffers {
        let vertices = [
            Vertex {
                position: [-1.0, -1.0],
            },
            Vertex {
                position: [-1.0, 3.0],
            },
            Vertex {
                position: [3.0, -1.0],
            },
        ];
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::empty()
                },
                false,
                vertices,
            )
            .expect("failed to create buffer")
        };

        Buffers { vertex_buffer }
    }

}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;

layout(location = 0) out vec2 v_frag_pos;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_frag_pos = position;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(push_constant) uniform PushConstants {
    vec3 light_position;
    vec3 light_intensity;
    float absorption;
    mat4 model_view;
    float focal_length;
    vec2 window_size;
    vec3 ray_origin;
} push_constants;

layout(set = 0, binding = 0) uniform sampler3D u_volume;

layout(location = 0) in vec2 v_frag_pos;
layout(location = 0) out vec4 f_color;

const float maxDist = sqrt(2.0);
const int numSamples = 128;
const float stepSize = maxDist/float(numSamples);
const int numLightSamples = 32;
const float lscale = maxDist / float(numLightSamples);
const float densityFactor = 5;

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

bool IntersectBox(Ray r, AABB aabb, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabb.Min-r.Origin);
    vec3 ttop = invR * (aabb.Max-r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    t1 = min(t.x, t.y);
    return t0 <= t1;
}

void main()
{
    vec3 rayDirection;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / push_constants.window_size - 1.0;
    rayDirection.z = -push_constants.focal_length;
    rayDirection = (vec4(rayDirection, 0) * push_constants.model_view).xyz;
    vec3 origin = push_constants.ray_origin;
    Ray eye = Ray( push_constants.ray_origin, normalize(rayDirection) );
    AABB aabb = AABB(vec3(-1.0), vec3(+1.0));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;


    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

    vec3 pos = rayStart;
    vec3 step = normalize(rayStop-rayStart) * stepSize;
    float travel = distance(rayStop, rayStart);
    float T = 1.0;
    vec3 Lo = vec3(0.0);

    for (int i=0; i < numSamples && travel > 0.0; ++i, pos += step, travel -= stepSize) {

        float density = texture(u_volume, pos).x * densityFactor;
        if (density <= 0.0)
            continue;

        T *= 1.0-density*stepSize*push_constants.absorption;
        if (T <= 0.01)
            break;

        vec3 lightDir = normalize(push_constants.light_position-pos)*lscale;
        float Tl = 1.0;
        vec3 lpos = pos + lightDir;

        for (int s=0; s < numLightSamples; ++s) {
            float ld = texture(u_volume, lpos).x;
            Tl *= 1.0-push_constants.absorption*stepSize*ld;
            if (Tl <= 0.01) 
            lpos += lightDir;
        }

        vec3 Li = push_constants.light_intensity*Tl;
        Lo += Li*T*density*stepSize;
    }

    f_color.rgb = Lo;
    f_color.a = 1-T;
}
",
    types_meta: {
        use bytemuck::{Pod, Zeroable};

        #[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
    },
}
}
