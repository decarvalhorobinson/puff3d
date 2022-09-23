use bytemuck::{Pod, Zeroable};
use cgmath::Matrix4;
use std::sync::Arc;
use std::sync::Mutex;
use vulkano::image::ImageViewAbstract;
use vulkano::impl_vertex;
use vulkano::pipeline::graphics::color_blend::AttachmentBlend;
use vulkano::pipeline::graphics::color_blend::BlendFactor;
use vulkano::pipeline::graphics::color_blend::BlendOp;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::render_pass::RenderPass;
use vulkano::sampler::BorderColor;
use vulkano::sampler::Filter;
use vulkano::sampler::Sampler;
use vulkano::sampler::SamplerAddressMode;
use vulkano::sampler::SamplerCreateInfo;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        SecondaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Queue,
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass,
};

use crate::scene_pkg::directional_light::DirectionalLight;

#[repr(C)]
#[derive(Clone)]
struct Buffers {
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

pub struct LightingPass {
    gfx_queue: Arc<Queue>,
    dir_light: Arc<Mutex<DirectionalLight>>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    buffers: Buffers,
}

impl LightingPass {
    /// Initializes a triangle drawing system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        dir_light: Arc<Mutex<DirectionalLight>>,
    ) -> LightingPass {
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let buffers = LightingPass::create_buffers(gfx_queue.clone());

        let pipeline = LightingPass::create_pipeline(gfx_queue.clone(), subpass.clone());

        LightingPass {
            gfx_queue,
            dir_light,
            pipeline,
            subpass,
            buffers,
        }
    }

    pub fn draw(
        &mut self,
        viewport_dimensions: [u32; 2],
        world: Matrix4<f32>,
        view: Matrix4<f32>,
        shadow_image: Arc<dyn ImageViewAbstract + 'static>,
        position_image: Arc<dyn ImageViewAbstract + 'static>,
        color_image: Arc<dyn ImageViewAbstract + 'static>,
        normals_image: Arc<dyn ImageViewAbstract + 'static>,
    ) -> SecondaryAutoCommandBuffer {
        let (light_view, light_projection) =
            self.dir_light.lock().unwrap().clone().view_projection();

        let push_constants_fs;
        {
            let dir_light_locked = self.dir_light.lock().unwrap();
            push_constants_fs = fs::ty::PushConstants {
                color: dir_light_locked.color,
                direction: dir_light_locked.clone().direction().extend(0.0).into(),
                light_proj_view_model: (light_projection * light_view).into(),
                world: world.into(),
                view: view.into(),
            };
        }

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                LightingPass::create_shadow_image_set(
                    self.gfx_queue.clone(),
                    0,
                    shadow_image.clone(),
                ),
                LightingPass::create_image_set(self.gfx_queue.clone(), 1, position_image.clone()),
                LightingPass::create_image_set(self.gfx_queue.clone(), 2, color_image.clone()),
                LightingPass::create_image_set(self.gfx_queue.clone(), 3, normals_image.clone()),
            ],
        )
        .unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let mut builder = AutoCommandBufferBuilder::secondary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
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
                descriptor_set,
            )
            .push_constants(self.pipeline.layout().clone(), 0, push_constants_fs)
            .bind_vertex_buffers(0, self.buffers.vertex_buffer.clone())
            .draw(self.buffers.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    // private methods

    fn create_image_set(
        gfx_queue: Arc<Queue>,
        binding_index: u32,
        image_view: Arc<dyn ImageViewAbstract + 'static>,
    ) -> WriteDescriptorSet {
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        WriteDescriptorSet::image_view_sampler(binding_index, image_view, sampler)
    }

    fn create_shadow_image_set(
        gfx_queue: Arc<Queue>,
        binding_index: u32,
        image_view: Arc<dyn ImageViewAbstract + 'static>,
    ) -> WriteDescriptorSet {
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                border_color: BorderColor::FloatOpaqueWhite,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                ..Default::default()
            },
        )
        .unwrap();

        WriteDescriptorSet::image_view_sampler(binding_index, image_view, sampler)
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

    fn create_buffers(gfx_queue: Arc<Queue>) -> Buffers {
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



layout(set = 0, binding = 0) uniform sampler2D shadow_map;

// The `position_input` parameter of the `draw` method.
layout(set = 0, binding = 1) uniform sampler2D u_position;
// The `color_input` parameter of the `draw` method.
layout(set = 0, binding = 2) uniform sampler2D u_diffuse;
// The `normals_input` parameter of the `draw` method.
layout(set = 0, binding = 3) uniform sampler2D u_normals;

layout(push_constant) uniform PushConstants {
    vec4 color;
    vec4 direction;
    mat4 light_proj_view_model;
    mat4 world;
    mat4 view;
} push_constants;

layout(location = 0) in vec2 v_frag_pos;
layout(location = 0) out vec4 f_color;

float shadow_calculation(float light_percent) {
    vec2 coord = v_frag_pos * 0.5 + 0.5;

    vec4 in_position = texture(u_position, coord);

    vec4 position_to_light = push_constants.light_proj_view_model * push_constants.world * in_position;
    position_to_light.xy  = position_to_light.xy * 0.5 + 0.5;

    float depth = texture(shadow_map, position_to_light.xy).r;

    if (position_to_light.z > 1.0) {
        return 0.0;
    }

    float bias = max(0.002 * (1.0 + light_percent), 0.0005); 
    
    
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcf_depth = texture(shadow_map, position_to_light.xy + vec2(x, y) * texel_size).r; 
            shadow += position_to_light.z - bias > pcf_depth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    return shadow;
}


void main() {
    vec2 coord = v_frag_pos * 0.5 + 0.5;
    vec4 in_position = texture(u_position, coord);
    vec3 in_diffuse = texture(u_diffuse, coord).rgb;
    vec3 in_normal = normalize(texture(u_normals, coord).rgb);
    vec4 in_shadow_map = texture(shadow_map, coord);


    vec4 light_to_world = normalize(inverse(push_constants.view) * push_constants.direction);
    float light_percent = -dot(light_to_world.xyz, in_normal);
    light_percent = max(light_percent, 0.3);

    float shadow = shadow_calculation(light_percent);
    vec3 ambient_light = (vec3(0.2, 0.2, 0.2) * in_diffuse);
    f_color.rgb = push_constants.color.rgb * in_diffuse * (1-shadow) * light_percent + ambient_light;
    f_color.a = 1.0;
    
}",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
