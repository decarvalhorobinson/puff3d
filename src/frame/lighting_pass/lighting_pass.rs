use bytemuck::{Pod, Zeroable};
use cgmath::Matrix4;
use cgmath::Vector4;
use vulkano::buffer::BufferAccess;
use vulkano::buffer::CpuBufferPool;
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
    lights_buffer:  CpuBufferPool::<fs::ty::Lights>
}

pub struct LightingPass {
    gfx_queue: Arc<Queue>,
    dir_lights: Vec<Arc<Mutex<DirectionalLight>>>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    buffers: Buffers,
}

impl LightingPass {
    /// Initializes a triangle drawing system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        dir_lights: Vec<Arc<Mutex<DirectionalLight>>>,
    ) -> LightingPass {
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let buffers = LightingPass::create_buffers(gfx_queue.clone());

        let pipeline = LightingPass::create_pipeline(gfx_queue.clone(), subpass.clone());

        LightingPass {
            gfx_queue,
            dir_lights,
            pipeline,
            subpass,
            buffers,
        }
    }

    pub fn draw(
        &mut self,
        viewport_dimensions: [u32; 2],
        camera_pos: Vector4<f32>,
        world: Matrix4<f32>,
        view: Matrix4<f32>,
        shadow_image: Arc<dyn ImageViewAbstract + 'static>,
        position_image: Arc<dyn ImageViewAbstract + 'static>,
        color_image: Arc<dyn ImageViewAbstract + 'static>,
        normals_image: Arc<dyn ImageViewAbstract + 'static>,
        metallic_image: Arc<dyn ImageViewAbstract + 'static>,
        roughness_image: Arc<dyn ImageViewAbstract + 'static>,
        ao_image: Arc<dyn ImageViewAbstract + 'static>,
        volume_image: Arc<dyn ImageViewAbstract + 'static>,
    ) -> SecondaryAutoCommandBuffer {
        let (light_view, light_projection) = self.dir_lights[0].lock().unwrap().clone().view_projection();
        let push_constants_fs;
        {
            let dir_light_locked = self.dir_lights[0].lock().unwrap();
            push_constants_fs = fs::ty::PushConstants {
                color: dir_light_locked.color,
                camera_pos: camera_pos.into(),
                light_pos: dir_light_locked.position.to_homogeneous().into(),
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
                LightingPass::create_image_set(self.gfx_queue.clone(), 4, metallic_image.clone()),
                LightingPass::create_image_set(self.gfx_queue.clone(), 5, roughness_image.clone()),
                LightingPass::create_image_set(self.gfx_queue.clone(), 6, ao_image.clone()),
                LightingPass::create_image_set(self.gfx_queue.clone(), 7, volume_image.clone()),
                self.create_lights_set()
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
                descriptor_set,
            )
            .push_constants(self.pipeline.layout().clone(), 0, push_constants_fs)
            
            .bind_vertex_buffers(0, self.buffers.vertex_buffer.clone())
            .draw(self.buffers.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    // private methods

    fn transform_lights(&self) {

    }

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

    fn create_lights_set(&self) -> WriteDescriptorSet {

        let mut shader_lights: [fs::ty::DirectionalLight; 4]= Default::default();
        for i in 0..shader_lights.len()  {
            let pos = self.dir_lights[i].lock().unwrap().position.into();
            let shader_light = fs::ty::DirectionalLight {
                position: pos,
                _dummy0: [1,1,1,1],
            };
            shader_lights[i] = shader_light;
            
        }
        //descriptor set
        let uniform_buffer_subbuffer = {
            let uniform_data = fs::ty::Lights {
                dir_lights: shader_lights
            };

            self.buffers.lights_buffer.from_data(uniform_data).unwrap()
        };

        WriteDescriptorSet::buffer(8, uniform_buffer_subbuffer)
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

        let lights_buffer = CpuBufferPool::<fs::ty::Lights>::new(
            gfx_queue.device().clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
        );

        Buffers { vertex_buffer, lights_buffer}
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

layout(set = 0, binding = 1) uniform sampler2D u_position;
layout(set = 0, binding = 2) uniform sampler2D u_base_color;
layout(set = 0, binding = 3) uniform sampler2D u_normals;
layout(set = 0, binding = 4) uniform sampler2D u_metallic;
layout(set = 0, binding = 5) uniform sampler2D u_roughness;
layout(set = 0, binding = 6) uniform sampler2D u_ao;
layout(set = 0, binding = 7) uniform sampler2D u_volume;
struct DirectionalLight {
    vec3 position;
};

layout(set = 0, binding = 8) uniform Lights {
    DirectionalLight[4] dir_lights;
} lights;

layout(push_constant) uniform PushConstants {
    vec4 color;
    vec4 camera_pos;
    vec4 light_pos;
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

    float bias = max(0.005 * (1.0 + light_percent), 0.001); 
    
    
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

// PBR methods
const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float distribution_ggx(vec3 n, vec3 h, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float n_dot_h = max(dot(n, h), 0.0);
    float n_dot_h2 = n_dot_h*n_dot_h;

    float nom   = a2;
    float denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float geometry_schlick_ggx(float n_dot_v, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = n_dot_v;
    float denom = n_dot_v * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness)
{
    float n_dot_v = max(dot(n, v), 0.0);
    float n_dot_l = max(dot(n, l), 0.0);
    float ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    float ggx1 = geometry_schlick_ggx(n_dot_l, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnel_schlick(float cos_theta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------
// PBR methods end


vec4 light() {
    vec2 coord = v_frag_pos * 0.5 + 0.5;
    vec4 in_position = texture(u_position, coord);
    vec3 in_base_color = pow(texture(u_base_color, coord).rgb, vec3(2.2));
    vec3 in_normal = normalize(texture(u_normals, coord).rgb);
    vec4 in_shadow_map = texture(shadow_map, coord);
    vec3 in_metallic = texture(u_metallic, coord).rgb;
    vec3 in_roughness = texture(u_roughness, coord).rgb;
    vec3 in_ao = texture(u_ao, coord).rgb;
    

    float metallic = in_metallic.r;
    float roughness = in_roughness.r;
    float ao = in_ao.r;

    in_normal = in_normal * 2 - 1;

    vec3 N = normalize(in_normal);
    vec3 V = normalize(push_constants.camera_pos).xyz;

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the in_base_color color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, in_base_color, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) 
    {

        // calculate per-light radiance
        vec3 L = normalize(lights.dir_lights[i].position);
        vec3 H = normalize(V + L);
        float distance = length(lights.dir_lights[i].position);
        float attenuation = 1.0 / (distance*distance);//(1 * 0.06*distance + 0.05*(distance * distance));
        vec3 light_color = vec3(1.0, 1.0, 1.0);
        vec3 radiance = light_color * attenuation;

        // Cook-Torrance BRDF
        float NDF = distribution_ggx(N, H, roughness);   
        float G   = geometry_smith(N, V, L, roughness);      
        vec3 F    = fresnel_schlick(clamp(dot(H, V), 0.0, 1.0), F0);
            
        vec3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        vec3 specular = numerator / denominator;
        
        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;	  

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * in_base_color / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }
    // ambient lighting (note that the next IBL tutorial will replace 
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * in_base_color * ao;

    vec3 color = ambient + Lo;

    // HDR tonemapping
    //color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    return vec4(color, 1.0);
}

void main() {
    vec2 coord = v_frag_pos * 0.5 + 0.5;
    vec4 in_position = texture(u_position, coord);
    vec3 in_base_color = texture(u_base_color, coord).rgb;
    vec3 in_normal = normalize(texture(u_normals, coord).rgb);
    vec4 in_shadow_map = texture(shadow_map, coord);
    vec3 in_metallic = texture(u_metallic, coord).rgb;
    vec3 in_roughness = texture(u_roughness, coord).rgb;
    vec3 in_ao = texture(u_ao, coord).rgb;
    vec3 in_volume = texture(u_volume, coord).rgb;

    in_normal = in_normal * 2 - 1;
    vec4 light_to_world = normalize(push_constants.direction);
    float radiance = -dot(light_to_world.xyz, in_normal);
    radiance = max(radiance, 0.1);

    float shadow = shadow_calculation(radiance);

    f_color = light();// * (1 - shadow);

    f_color.rgb = in_volume;

    
}",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
        },
    }
}
