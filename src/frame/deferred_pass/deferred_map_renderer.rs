use std::sync::{Arc, Mutex};
use vulkano::{sync::{self, GpuFuture}, image::{AttachmentImage, ImageUsage}, format::Format};

use cgmath::{Matrix4, SquareMatrix};
use image::{ImageBuffer, Rgba};
use vulkano::{device::Queue, render_pass::{Subpass, RenderPass, Framebuffer, FramebufferCreateInfo}, command_buffer::{PrimaryAutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents, SecondaryCommandBuffer, CopyImageToBufferInfo}, image::{ImageViewAbstract, StorageImage, view::ImageView}, buffer::{CpuAccessibleBuffer, BufferUsage}};

use crate::scene_pkg::scene::Scene;

use super::object_3d_deferred_pass::Object3DDeferredPass;

pub struct DeferredMapRenderer {

    pub scene: Arc<Mutex<Scene>>,
    pub gfx_queue: Arc<Queue>,
    pub render_pass: Arc<RenderPass>,
    pub object_3d_passes: Vec<Object3DDeferredPass>,

    pub framebuffer: Option<Arc<Framebuffer>>,
    pub command_buffer_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,

    pub position_buffer: Arc<ImageView<AttachmentImage>>,
    pub albedo_specular_buffer: Arc<ImageView<AttachmentImage>>,
    pub normals_buffer: Arc<ImageView<AttachmentImage>>,
    depth_buffer: Arc<ImageView<AttachmentImage>>,
    
}

impl DeferredMapRenderer {

    pub fn new(gfx_queue: Arc<Queue>, scene: Arc<Mutex<Scene>>) -> DeferredMapRenderer {
        let render_pass = vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                // Will be bound to `self.position_buffer`.
                position: {
                    load: Clear,
                    store: Store,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                },
                // Will be bound to `self.albedo_specular_buffer`.
                albedo_specular: {
                    load: Clear,
                    store: Store,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                },
                // Will be bound to `self.normals_buffer`.
                normals: {
                    load: Clear,
                    store: Store,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                },
                // Will be bound to `self.depth_buffer`.
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            passes: [
                // Write to the diffuse, normals and depth attachments.
                {
                    color: [position, albedo_specular, normals],
                    depth_stencil: {depth},
                    input: []
                }
            ]
        )
        .unwrap();

        let position_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1024, 1024],
                Format::R16G16B16A16_SFLOAT,
                ImageUsage {
                    transient_attachment: false,
                    input_attachment: false,
                    sampled: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let albedo_specular_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1024, 1024],
                Format::R16G16B16A16_SFLOAT,
                ImageUsage {
                    transient_attachment: false,
                    input_attachment: false,
                    sampled: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();
        let normals_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1024, 1024],
                Format::R16G16B16A16_SFLOAT,
                ImageUsage {
                    transient_attachment: false,
                    input_attachment: false,
                    sampled: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();
        let depth_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1024, 1024],
                Format::D16_UNORM,
                ImageUsage {
                    transient_attachment: true,
                    input_attachment: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let scene_locked = scene.lock().unwrap();
        let mut object_3d_passes: Vec<Object3DDeferredPass> = vec![];
        object_3d_passes.reserve(scene_locked.objects.len());
        for object_3d in scene_locked.objects.clone()  {
            object_3d_passes.push(Object3DDeferredPass::new(
                gfx_queue.clone(),
                render_pass.clone(),
                object_3d
            ));
        }

        DeferredMapRenderer { 
            scene: scene.clone(),
            gfx_queue: gfx_queue.clone(),
            render_pass: render_pass,
            object_3d_passes: object_3d_passes,
            framebuffer: Option::None,
            command_buffer_builder: Option::None,

            position_buffer: position_buffer,
            albedo_specular_buffer: albedo_specular_buffer,
            normals_buffer: normals_buffer,
            depth_buffer: depth_buffer
            

        }
    }

    pub fn draw(&mut self) {
        let projection = Scene::projection(self.viewport_dimensions());
        let view;
        let world;
        {
            let scene_locked = self.scene.lock().unwrap();
            world = scene_locked.world_model;
            view = scene_locked.active_camera.get_view_matrix();
        }

        for i in 0..self.object_3d_passes.len() {

            let cb = self.object_3d_passes[i].draw(self.framebuffer.clone().unwrap().extent(), world,  projection, view);
            self.execute_draw_pass(cb);
        }


    }


    pub fn begin_render_pass(
        &mut self
    )
    {
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![self.position_buffer.clone(), self.albedo_specular_buffer.clone(), self.normals_buffer.clone(), self.depth_buffer.clone()],
                ..Default::default()
            },
        )
        .unwrap();
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some(1.0f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap();
        self.framebuffer = Some(framebuffer);
        self.command_buffer_builder= Some(command_buffer_builder);

    }

    pub fn execute_draw_pass<C>(&mut self, command_buffer: C)
    where
        C: SecondaryCommandBuffer + 'static,
    {
        self.command_buffer_builder
            .as_mut()
            .unwrap()
            .execute_commands(command_buffer)
            .unwrap();
    }

    pub fn end_render_pass<F: GpuFuture + 'static>(&mut self, future: F) -> vulkano::command_buffer::CommandBufferExecFuture<F, PrimaryAutoCommandBuffer>  {

        self.command_buffer_builder
            .as_mut()
            .unwrap()
            .end_render_pass()
            .unwrap();
        let command_buffer = self.command_buffer_builder.take().unwrap().build().unwrap();

        future.then_execute(self.gfx_queue.clone(), command_buffer).unwrap()



    }
    
    #[inline]
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        self.framebuffer.clone().unwrap().extent()
    }
    
}