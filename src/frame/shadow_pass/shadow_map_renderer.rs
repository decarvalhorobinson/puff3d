use std::sync::{Arc, Mutex};
use vulkano::{sync::{self, GpuFuture}, image::{AttachmentImage, ImageUsage}, format::Format};

use cgmath::{Matrix4, SquareMatrix};
use image::{ImageBuffer, Rgba};
use vulkano::{device::Queue, render_pass::{Subpass, RenderPass, Framebuffer, FramebufferCreateInfo}, command_buffer::{PrimaryAutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents, SecondaryCommandBuffer, CopyImageToBufferInfo}, image::{ImageViewAbstract, StorageImage, view::ImageView}, buffer::{CpuAccessibleBuffer, BufferUsage}};

use crate::scene_pkg::scene::Scene;

use super::object_3d_shadow_pass::Object3DShadowPass;

pub struct ShadowMapRenderer {

    pub scene: Arc<Mutex<Scene>>,
    pub gfx_queue: Arc<Queue>,
    pub render_pass: Arc<RenderPass>,
    pub object_3d_passes: Vec<Object3DShadowPass>,

    pub framebuffer: Option<Arc<Framebuffer>>,
    pub command_buffer_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,

    pub final_image: Arc<AttachmentImage>

}

impl ShadowMapRenderer {

    pub fn new(queue: Arc<Queue>, scene: Arc<Mutex<Scene>>) -> ShadowMapRenderer {
        let render_pass = vulkano::single_pass_renderpass!(queue.device().clone(),
                attachments: {
                    depth: {
                        load: Clear,
                        store: Store,
                        format: Format::D16_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [],
                    depth_stencil: {depth}
                }
            ).unwrap();

        let shadow_image = AttachmentImage::with_usage(
            queue.device().clone(),
            [2048, 2048],
            Format::D16_UNORM,
            ImageUsage {
                transient_attachment: false,
                input_attachment: false,
                sampled: true,
                ..ImageUsage::none()
            },
        )
        .unwrap();

        let scene_locked = scene.lock().unwrap();
        let mut object_3d_passes: Vec<Object3DShadowPass> = vec![];
        object_3d_passes.reserve(scene_locked.objects.len());
        for object_3d in scene_locked.objects.clone()  {
            object_3d_passes.push(Object3DShadowPass::new(
                queue.clone(),
                render_pass.clone(),
                object_3d
            ));
        }

        ShadowMapRenderer { 
            scene: scene.clone(),
            gfx_queue: queue.clone(),
            render_pass: render_pass,
            object_3d_passes: object_3d_passes,
            framebuffer: Option::None,
            command_buffer_builder: Option::None,
            final_image: shadow_image
        }
    }

    pub fn draw(&mut self) {
        let view;
        let projection;
        let world;
        {
            let scene_locked = self.scene.lock().unwrap();
            world = scene_locked.world_model;
            (view, projection) = scene_locked.directional_lights[0].clone().view_projection();
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
        let view = ImageView::new_default(self.final_image.clone()).unwrap();
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view],
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
                    clear_values: vec![Some(1.0f32.into())],
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