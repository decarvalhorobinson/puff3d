use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, SquareMatrix, Deg, Angle, Point3, Matrix, Vector3};
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
    volume: Volume,
    rotation: f32

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

        let volume_write_set = VolumePass::create_volume_set(pipeline.clone(), gfx_queue.clone(), volume.clone());

        let transfer_write_image_set = VolumePass::create_transfer_image(pipeline.clone(), gfx_queue.clone());

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let volume_set = PersistentDescriptorSet::new(
            layout.clone(),
            [volume_write_set, transfer_write_image_set],
        )
        .unwrap();

        

        VolumePass {
            gfx_queue,
            pipeline,
            subpass,
            buffers,
            volume,
            volume_set,
            rotation: 0.0
        }
    }

    pub fn draw(
        &mut self,
        viewport_dimensions: [u32; 2],
        world: Matrix4<f32>,
        view: Matrix4<f32>,
        eye_pos: Point3<f32>,
        delta_time: u128

    ) -> SecondaryAutoCommandBuffer {
        self.rotation += 0.05 * delta_time as f32;
        self.volume.transform.rotation = Vector3::new(90.0, self.rotation, 00.0);
        self.volume.update_model_matrix();
        let focal_length = 1.0 / Deg(72.0 / 2.0).tan();
        let push_constants_fs;
        {
            let ray_origin =  eye_pos.to_homogeneous();
            push_constants_fs = fs::ty::PushConstants {
                light_position: [-16.0, 0.0, 6.0], ///[3.0, 10.0, 3.0],[3.0, 10.0, 3.0]
                light_intensity: [15.5, 15.5, 15.5],
                absorption: 1.0,
                model_view: (view).into(),
                model: self.volume.model_matrix.into(),
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

    fn create_transfer_image(pipeline: Arc<GraphicsPipeline>, gfx_queue: Arc<Queue>) -> WriteDescriptorSet {
        let dim2 = ImageDimensions::Dim2d { width: 1000, height: 1000, array_layers: 1 };
        let mut pixel_data: Vec<u8> = vec![];

        let test = 900;
        let factor = 5;

        for h in 0..dim2.height()  {
    
            for w in 0..dim2.width() {
                let mut red = 0;
                let mut green = 0;
                let mut blue = 0;
                let mut alpha = 0;

                if w > test && w <= test + factor{
                    red = 255;
                    alpha = 150;
                } else if w > test + factor * 2 && w <= test + factor * 3 {
                    //red = 255;
                    //alpha = 150;
                } else if w > test + factor * 4 && w <= test + factor * 5 {
                    //red = 255;
                    //alpha = 150;
                }

                pixel_data.push(red);
                pixel_data.push(green);
                pixel_data.push(blue);
                pixel_data.push(alpha);
            }
        }

        let (image, future) = ImmutableImage::from_iter(
            pixel_data,
            dim2,
            MipmapsCount::One,
            Format::R8G8B8A8_UNORM,
            gfx_queue.clone(),
        )
        .unwrap();
        let image_view = ImageView::new_default(image).unwrap();
        
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                ..Default::default()
            },
        )
        .unwrap();

        let write_descriptor = WriteDescriptorSet::image_view_sampler(1, image_view, sampler);

        write_descriptor

    }

    fn create_volume_set(pipeline: Arc<GraphicsPipeline>, gfx_queue: Arc<Queue>, volume: Volume) -> WriteDescriptorSet {
        
        let dim3 = ImageDimensions::Dim3d { width: volume.dimension[0], height: volume.dimension[1], depth: volume.dimension[2] };
        let (image, future) = ImmutableImage::from_iter(
            volume.pixel_data,
            dim3,
            MipmapsCount::One,
            Format::R16_UNORM,
            gfx_queue.clone(),
        )
        .unwrap();
        let image_view = ImageView::new_default(image).unwrap();
        
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                ..Default::default()
            },
        )
        .unwrap();

        let write_descriptor = WriteDescriptorSet::image_view_sampler(0, image_view, sampler);

        

        write_descriptor
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
    mat4 model;
    float focal_length;
    vec2 window_size;
    vec3 ray_origin;
} push_constants;

layout(set = 0, binding = 0) uniform sampler3D u_volume;
layout(set = 0, binding = 1) uniform sampler2D u_transfer_image;

layout(location = 0) in vec2 v_frag_pos;
layout(location = 0) out vec4 f_color;

const float maxDist = sqrt(1.0);
const int numSamples = 500;
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

bool render_x_ray_render(inout vec3 pos, inout float T, inout vec3 Lo) {
    float density = texture(u_volume, pos).x * densityFactor;
    if (density <= 0.0)
        return false;

    T *= 1.0-density*stepSize*push_constants.absorption;
    if (T <= 0.01)
        return true;

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

    return false;
}

bool max_intensity_render(inout vec3 pos, inout float T, inout vec3 Lo, inout float max_intensity) {
    float density = texture(u_volume, pos).x * densityFactor;
    if (density > max_intensity) {
        max_intensity = density;
        Lo = push_constants.light_intensity * max_intensity;
    }

    return false;
}


const vec3 lightPos = vec3(1.0, 1.0, 1.0);
const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const float lightPower = 40.0;
const vec3 ambientColor = vec3(0.1, 0.0, 0.0);
const vec3 diffuseColor = vec3(0.5, 0.0, 0.0);
const vec3 specColor = vec3(1.0, 1.0, 1.0);
const float shininess = 130.0;

vec4 transfer_color(float signal_intensity, float gradient_magnitude) {
    if (signal_intensity <= 0.0)
    {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    signal_intensity *= 20; 
    vec4 color = texture(u_transfer_image, vec2(signal_intensity, 0.0));

    return color;
}

bool vr_render(inout vec3 pos, inout float T, inout vec3 Lo, inout vec3 normal, vec3 ray, vec3 step) {
    float density = texture(u_volume, pos).x * densityFactor;
    vec4 material_colour = transfer_color(texture(u_volume, pos).x, 0);
    if (material_colour.a < 0.0000000005)
    {
        return false;
    }

    vec3 lightDir = normalize(push_constants.light_position-pos)*lscale;

    //calculate normals
    float pos_relative = 1.0/512.0;
    float density_x_plus = texture(u_volume, pos + vec3(pos_relative, 0.0, 0.0)).r;
    float density_x_minus = texture(u_volume, pos - vec3(pos_relative, 0.0, 0.0)).r;

    float density_y_plus = texture(u_volume, pos + vec3(0.0, pos_relative, 0.0)).r;
    float density_y_minus = texture(u_volume, pos - vec3(0.0, pos_relative, 0.0)).r;

    float density_z_plus = texture(u_volume, pos + vec3(0.0, 0.0, pos_relative)).r;
    float density_z_minus = texture(u_volume, pos - vec3(0.0, 0.0, pos_relative)).r;


    float gx = density_x_plus - density_x_minus;
    float gy = density_y_plus - density_y_minus;
    float gz = density_z_plus - density_z_minus;

    normal = normalize(vec3(gx, gy, gz));    

    // Blinn-Phong shading
    vec3 L = -normalize(push_constants.light_position - pos);
    //L = (push_constants.model * vec4(L,1.0)).xyz;
    vec3 V = -normalize(ray);
    vec3 N = normal;
    N = (inverse(push_constants.model) * vec4(N,1.0)).xyz;
    vec3 H = normalize(L + V);
    

    float Ia = 0.1;
    float Id = 1.0 * max(0, dot(N, L));
    float Is = 8.0 * pow(max(0, dot(N, H)), 600);
    Lo += ((Ia + Id) * material_colour.xyz + Is * vec3(1.0)) * material_colour.a;
    
    T -= material_colour.a;
    if (T <= 0.0)
    {
        return true;
    } else {
        return false;
    }
    

    
}

bool vr_render_bkp(inout vec3 pos, inout float T, inout vec3 Lo, inout vec3 normal, vec3 ray, vec3 step) {
    float density = texture(u_volume, pos).x * densityFactor;
    float threshold = 0.095;
    if (density < threshold)
        return false;

    
    // Get closer to the surface
    pos -= step * 0.5;
    density = texture(u_volume, pos).r * densityFactor;
    pos -= step * (density > threshold ? 0.25 : -0.25);
    density = texture(u_volume, pos).r * densityFactor;
    pos -= step * (density > threshold ? 0.25 : -0.25);
    density = texture(u_volume, pos).r * densityFactor;

    vec3 lightDir = normalize(push_constants.light_position-pos)*lscale;

    //calculate normals
    float pos_relative = 1.0/512.0;
    float density_x_plus = texture(u_volume, pos + vec3(pos_relative, 0.0, 0.0)).r;
    float density_x_minus = texture(u_volume, pos - vec3(pos_relative, 0.0, 0.0)).r;

    float density_y_plus = texture(u_volume, pos + vec3(0.0, pos_relative, 0.0)).r;
    float density_y_minus = texture(u_volume, pos - vec3(0.0, pos_relative, 0.0)).r;

    float density_z_plus = texture(u_volume, pos + vec3(0.0, 0.0, pos_relative)).r;
    float density_z_minus = texture(u_volume, pos - vec3(0.0, 0.0, pos_relative)).r;


    float gx = density_x_plus - density_x_minus;
    float gy = density_y_plus - density_y_minus;
    float gz = density_z_plus - density_z_minus;

    normal = normalize(vec3(gx, gy, gz));    

    //density = ((density_x_plus + density_x_minus)/2 + (density_y_plus + density_y_minus)/2 + (density_z_plus + density_z_minus)/2)/3 * densityFactor;


    // Blinn-Phong shading
    vec3 L = -normalize(push_constants.light_position - pos);
    //L = (push_constants.model * vec4(L,1.0)).xyz;
    vec3 V = -normalize(ray);
    vec3 N = normal;
    N = (inverse(push_constants.model) * vec4(N,1.0)).xyz;
    vec3 H = normalize(L + V);
    vec3 material_colour = transfer_color(texture(u_volume, pos).x, 0).xyz;

    float Ia = 0.1;
    float Id = 1.0 * max(0, dot(N, L));
    float Is = 8.0 * pow(max(0, dot(N, H)), 600);
    Lo = (Ia + Id) * material_colour + Is * vec3(1.0);
    
    T = 0.0;
    return true;

    
}

float get_density_from_volume(vec3 pos) {
    vec3 real_pos = pos * 512.0;
    float density = texture(u_volume, pos).x * densityFactor; 
    return density;
}

void main()
{
    
    float random = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 rayDirection;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / push_constants.window_size - 1.0;
    rayDirection.z = -push_constants.focal_length;
    rayDirection = (vec4(rayDirection, 0) * push_constants.model_view).xyz;

    vec3 origin = push_constants.ray_origin;

    Ray eye = Ray( origin, normalize(rayDirection) );
    AABB aabb = AABB(vec3(-1.0), vec3(+1.0));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    
    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    rayStart = (push_constants.model * vec4(rayStart, 1.0)).xyz;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStop = (push_constants.model * vec4(rayStop, 1.0)).xyz;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);
    
    vec3 pos = rayStart + stepSize * random * 0.5;
    vec3 step = normalize(rayStop-rayStart) * stepSize ;
    float travel = distance(rayStop, rayStart);
    float T = 1.0;
    vec3 Lo = vec3(0.0);
    float max_intensity = 0.0;
    vec3 normal = vec3(0.0, 1.0, 0.0);

    for (int i=0; i < numSamples && travel > 0.0; ++i, pos += step, travel -= stepSize) {
        //bool can_break_for = render_x_ray_render(pos, T, Lo);
        //bool can_break_for = max_intensity_render(pos, T, Lo, max_intensity);
        bool can_break_for = vr_render(pos, T, Lo, normal, eye.Origin, step);
        if(can_break_for) {
            break;
        }
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
