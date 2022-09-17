// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::{Matrix3, Rad, Matrix4, Point3, Vector3};
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use std::{sync::Arc, io::Cursor};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess, CpuBufferPool},
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
    render_pass::Subpass, image::{ImageDimensions, ImmutableImage, MipmapsCount, view::ImageView}, format::Format, sampler::{Sampler, SamplerCreateInfo, SamplerAddressMode, Filter},
};

use crate::scene_pkg::mesh::{ Normal, Tangent, Uv, Vertex };
use crate::scene_pkg::object3d::Object3D;

use super::shadow_map_renderer::ShadowMapRenderer;

pub struct Object3DShadowPass{
    object_3d: Object3D,
    pipeline_depth: Arc<GraphicsPipeline>,
    subpass: Subpass, 
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>, 
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>, 
    uniform_data_buffer: CpuBufferPool<vs_depth::ty::Data>
}

impl Object3DShadowPass{
    /// Initializes a triangle drawing system.
    pub fn new(
        shadow_map_renderer: &ShadowMapRenderer, 
        object_3d: Object3D
    ) -> Object3DShadowPass {

        let subpass = Subpass::from(shadow_map_renderer.render_pass.clone(), 0).unwrap();

        let (vertex_buffer, index_buffer, uniform_data_buffer) = 
            Object3DShadowPass::create_buffers(shadow_map_renderer.gfx_queue.clone(), object_3d.clone());

        let pipeline_depth = Object3DShadowPass::create_depth_pipeline(shadow_map_renderer.gfx_queue.clone(), subpass.clone());

        Object3DShadowPass {
            object_3d,
            pipeline_depth,
            subpass,
            vertex_buffer,
            index_buffer,
            uniform_data_buffer
        }
    }


    pub fn draw(&mut self, shadow_map_renderer: &ShadowMapRenderer,  world: Matrix4<f32>, projection: Matrix4<f32>, view: Matrix4<f32>) -> SecondaryAutoCommandBuffer {

        let viewport_dimensions = shadow_map_renderer.viewport_dimensions();
        //descriptor set
        let uniform_buffer_subbuffer = {
            let scale = Matrix4::from_scale(0.05);

            let uniform_data = vs_depth::ty::Data {
                model: self.object_3d.model_matrix.into(),
                world: world.into(),
                view: (view * scale).into(),
                proj: projection.into(),
            };

            self.uniform_data_buffer.next(uniform_data).unwrap()
        };

        let layout = self.pipeline_depth.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
        )
        .unwrap();

        //end descriptor set

        let mut builder = AutoCommandBufferBuilder::secondary(
            shadow_map_renderer.gfx_queue.device().clone(),
            shadow_map_renderer.gfx_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .set_viewport(
                0,
                [Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_pipeline_graphics(self.pipeline_depth.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline_depth.layout().clone(),
                0,
                set,

            )
            .bind_vertex_buffers(0, (self.vertex_buffer.clone()))
            .bind_index_buffer(self.index_buffer.clone())
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    // private methods

    fn create_depth_pipeline(gfx_queue: Arc<Queue>, subpass: Subpass) -> Arc<GraphicsPipeline> {
        let vs = vs_depth::load(gfx_queue.device().clone()).expect("failed to create shader module");
        let fs = fs_depth::load(gfx_queue.device().clone()).expect("failed to create shader module");

        GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>(),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(subpass.clone())
            .build(gfx_queue.device().clone())
            .unwrap()
    }

    fn create_buffers(gfx_queue: Arc<Queue>,  object_3d: Object3D) -> (Arc<CpuAccessibleBuffer<[Vertex]>>, Arc<CpuAccessibleBuffer<[u16]>>, CpuBufferPool<vs_depth::ty::Data>) {
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::none()
                },
                false,
                object_3d.mesh.vertices.clone(),
            )
            .expect("failed to create buffer")
        };

        let index_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    index_buffer: true,
                    ..BufferUsage::none()
                },
                false,
                object_3d.mesh.indices.clone(),
            )
            .expect("failed to create buffer")
        };

        let uniform_buffer_depth = CpuBufferPool::<vs_depth::ty::Data>::new(
            gfx_queue.device().clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
        );

        (vertex_buffer, index_buffer, uniform_buffer_depth)

    }
}

mod vs_depth {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform Data {
    mat4 model;
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world * uniforms.model;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);

}",
types_meta: {
    use bytemuck::{Pod, Zeroable};

    #[derive(Clone, Copy, Zeroable, Pod)]
}
    }
}

mod fs_depth {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450


void main() {
}"
    }
}
