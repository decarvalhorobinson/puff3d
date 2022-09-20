// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use bytemuck::{Pod, Zeroable};
use cgmath::SquareMatrix;
use cgmath::{Vector3, Matrix4};
use vulkano::image::AttachmentImage;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use std::{sync::Arc, io::Cursor};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        SecondaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Queue,
    image::{ImageViewAbstract, StorageImage, view::{ImageView, ImageViewCreateInfo, ImageViewType}, ImageDimensions, ImmutableImage, MipmapsCount},
    impl_vertex,
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass, sampler::{Sampler, SamplerCreateInfo, Filter, SamplerAddressMode}, format::Format,
};

use crate::scene_pkg::scene::Scene;

/// Allows applying a directional light source to a scene.
pub struct DirectionalLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
}

impl DirectionalLightingSystem {
    /// Initializes the directional lighting system.
    pub fn new(gfx_queue: Arc<Queue>, subpass: Subpass) -> DirectionalLightingSystem {
        // TODO: vulkano doesn't allow us to draw without a vertex buffer, otherwise we could
        //       hard-code these values in the shader
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
                    ..BufferUsage::none()
                },
                false,
                vertices,
            )
            .expect("failed to create buffer")
        };

        let pipeline = {
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
        };

        DirectionalLightingSystem {
            gfx_queue,
            vertex_buffer,
            subpass,
            pipeline,
        }
    }

    /// Builds a secondary command buffer that applies directional lighting.
    ///
    /// This secondary command buffer will read `color_input` and `normals_input`, and multiply the
    /// color with `color` and the dot product of the `direction` with the normal.
    /// It then writes the output to the current framebuffer with additive blending (in other words
    /// the value will be added to the existing value in the framebuffer, and not replace the
    /// existing value).
    ///
    /// Since `normals_input` contains normals in world coordinates, `direction` should also be in
    /// world coordinates.
    ///
    /// - `viewport_dimensions` contains the dimensions of the current framebuffer.
    /// - `color_input` is an image containing the albedo of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `normals_input` is an image containing the normals of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `direction` is the direction of the light in world coordinates.
    /// - `color` is the color to apply.
    ///
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        position_input: Arc<dyn ImageViewAbstract + 'static>,
        color_input: Arc<dyn ImageViewAbstract + 'static>,
        normals_input: Arc<dyn ImageViewAbstract + 'static>,
        direction: Vector3<f32>,
        light_view: Matrix4<f32>,
        light_projection: Matrix4<f32>,
        world: Matrix4<f32>,
        world_view_model: Matrix4<f32>,
        color: [f32; 3],
        shadow_image: Arc<AttachmentImage>,
    ) -> SecondaryAutoCommandBuffer {
        let push_constants_fs = fs::ty::PushConstants {
            color: [color[0], color[1], color[2], 1.0],
            direction: direction.extend(0.0).into(),
            light_proj_view_model: (light_projection * light_view ).into(),
            world: world.into(),
            world_view_model: (world_view_model).into()
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                DirectionalLightingSystem::create_image_set(self.gfx_queue.clone(), 0, ImageView::new_default(shadow_image.clone()).unwrap()),
                DirectionalLightingSystem::create_image_set(self.gfx_queue.clone(), 1, position_input.clone()),
                DirectionalLightingSystem::create_image_set(self.gfx_queue.clone(), 2, color_input.clone()),
                DirectionalLightingSystem::create_image_set(self.gfx_queue.clone(), 3, normals_input.clone())
                
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
            self.gfx_queue.family(),
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
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    fn create_image_set(gfx_queue: Arc<Queue>, binding_index: u32, image_view: Arc<dyn ImageViewAbstract + 'static>,) -> WriteDescriptorSet {
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
    mat4 world_view_model;
} push_constants;





layout(location = 0) in vec2 v_frag_pos;
layout(location = 0) out vec4 f_color;


void main() {
    vec4 in_position = texture(u_position, v_frag_pos * 0.5 + 0.5);
    vec3 in_diffuse = texture(u_diffuse, v_frag_pos * 0.5 + 0.5).rgb;
    vec3 in_normal = normalize(texture(u_normals, v_frag_pos * 0.5 + 0.5).rgb);
    vec4 in_shadow_map = texture(shadow_map, v_frag_pos * 0.5 + 0.5);


    float light_percent = -dot(push_constants.direction.xyz, in_normal);
    light_percent = max(light_percent, 0.0);

    f_color.rgb = light_percent * push_constants.color.rgb * in_diffuse;
    f_color.a = 1.0;

    f_color =  in_position;
    f_color =  push_constants.light_proj_view_model * push_constants.world * f_color;

    float depth = texture(shadow_map, f_color.xy*0.5+0.5).r;

    if (f_color.z - 0.005 > depth) {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
    }else{
        f_color = vec4(1.0, 1.0, 1.0, 1.0);
    }

    //f_color = vec4(vec3(depth), 1.0);




    
}",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
