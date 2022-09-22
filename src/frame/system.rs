// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::scene_pkg::scene::Scene;

use super::{
    directional_lighting_system::DirectionalLightingSystem,
};
use cgmath::{Matrix4, SquareMatrix, Vector3};
use std::sync::Arc;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SecondaryCommandBuffer, SubpassContents,
    },
    device::Queue,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, ImageViewAbstract, StorageImage},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

/// System that contains the necessary facilities for rendering a single frame.
pub struct FrameSystem {
    // Queue to use to render everything.
    gfx_queue: Arc<Queue>,

    // Render pass used for the drawing. See the `new` method for the actual render pass content.
    // We need to keep it in `FrameSystem` because we may want to recreate the intermediate buffers
    // in of a change in the dimensions.
    pub render_pass: Arc<RenderPass>,

    position_buffer: Arc<ImageView<AttachmentImage>>,
    // Intermediate render target that will contain the albedo of each pixel of the scene.
    diffuse_buffer: Arc<ImageView<AttachmentImage>>,
    // Intermediate render target that will contain the normal vector in world coordinates of each
    // pixel of the scene.
    // The normal vector is the vector perpendicular to the surface of the object at this point.
    normals_buffer: Arc<ImageView<AttachmentImage>>,
    // Intermediate render target that will contain the depth of each pixel of the scene.
    // This is a traditional depth buffer. `0.0` means "near", and `1.0` means "far".
    depth_buffer: Arc<ImageView<AttachmentImage>>,

    // Will allow us to add a directional light to a scene during the second subpass.
    directional_lighting_system: DirectionalLightingSystem,
}

impl FrameSystem {
    pub fn new(gfx_queue: Arc<Queue>, final_output_format: Format) -> FrameSystem {
        let render_pass = vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                // The image that will contain the final rendering (in this example the swapchain
                // image, but it could be another image).
                final_color: {
                    load: Clear,
                    store: Store,
                    format: final_output_format,
                    samples: 1,
                },
                // Will be bound to `self.position_buffer`.
                position: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                },
                // Will be bound to `self.diffuse_buffer`.
                diffuse: {
                    load: Clear,
                    store: DontCare,
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    samples: 1,
                },
                // Will be bound to `self.normals_buffer`.
                normals: {
                    load: Clear,
                    store: DontCare,
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
                    color: [position, diffuse, normals],
                    depth_stencil: {depth},
                    input: []
                },
                // Apply lighting by reading these three attachments and writing to `final_color`.
                {
                    color: [final_color],
                    depth_stencil: {},
                    input: [position, diffuse, normals, depth]
                }
            ]
        )
        .unwrap();

        // For now we create three temporary images with a dimension of 1 by 1 pixel.
        // These images will be replaced the first time we call `frame()`.
        let position_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1, 1],
                Format::R16G16B16A16_UNORM,
                ImageUsage {
                    transient_attachment: true,
                    input_attachment: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();
        let diffuse_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1, 1],
                Format::A2B10G10R10_UNORM_PACK32,
                ImageUsage {
                    transient_attachment: true,
                    input_attachment: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();
        let normals_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1, 1],
                Format::R16G16B16A16_SFLOAT,
                ImageUsage {
                    transient_attachment: true,
                    input_attachment: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();
        let depth_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1, 1],
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

        // Initialize the three lighting systems.
        // Note that we need to pass to them the subpass where they will be executed.
        let lighting_subpass = Subpass::from(render_pass.clone(), 1).unwrap();
        let directional_lighting_system =
            DirectionalLightingSystem::new(gfx_queue.clone(), lighting_subpass.clone());

        FrameSystem {
            gfx_queue,
            render_pass,
            position_buffer,
            diffuse_buffer,
            normals_buffer,
            depth_buffer,
            directional_lighting_system,
        }
    }

    /// Starts drawing a new frame.
    ///
    /// - `before_future` is the future after which the main rendering should be executed.
    /// - `final_image` is the image we are going to draw to.
    /// - `world_to_framebuffer` is the matrix that will be used to convert from 3D coordinates in
    ///   the world into 2D coordinates on the framebuffer.
    ///
    pub fn frame<F>(
        &mut self,
        before_future: F,
        final_image: Arc<dyn ImageViewAbstract + 'static>,
        world_to_framebuffer: Matrix4<f32>,
        shadow_image: Arc<AttachmentImage>
    ) -> Frame
    where
        F: GpuFuture + 'static,
    {
        // First of all we recreate `self.diffuse_buffer`, `self.normals_buffer` and
        // `self.depth_buffer` if their dimensions doesn't match the dimensions of the final image.
        let img_dims = final_image.image().dimensions().width_height();
        if self.diffuse_buffer.image().dimensions().width_height() != img_dims {
            // Note that we create "transient" images here. This means that the content of the
            // image is only defined when within a render pass. In other words you can draw to
            // them in a subpass then read them in another subpass, but as soon as you leave the
            // render pass their content becomes undefined.
            
            self.position_buffer = ImageView::new_default(
                AttachmentImage::with_usage(
                    self.gfx_queue.device().clone(),
                    img_dims,
                    Format::R16G16B16A16_SFLOAT,
                    ImageUsage {
                        transient_attachment: true,
                        input_attachment: true,
                        ..ImageUsage::none()
                    },
                )
                .unwrap(),
            )
            .unwrap();
            self.diffuse_buffer = ImageView::new_default(
                AttachmentImage::with_usage(
                    self.gfx_queue.device().clone(),
                    img_dims,
                    Format::A2B10G10R10_UNORM_PACK32,
                    ImageUsage {
                        transient_attachment: true,
                        input_attachment: true,
                        ..ImageUsage::none()
                    },
                )
                .unwrap(),
            )
            .unwrap();
            self.normals_buffer = ImageView::new_default(
                AttachmentImage::with_usage(
                    self.gfx_queue.device().clone(),
                    img_dims,
                    Format::R16G16B16A16_SFLOAT,
                    ImageUsage {
                        transient_attachment: true,
                        input_attachment: true,
                        ..ImageUsage::none()
                    },
                )
                .unwrap(),
            )
            .unwrap();
            self.depth_buffer = ImageView::new_default(
                AttachmentImage::with_usage(
                    self.gfx_queue.device().clone(),
                    img_dims,
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
        }

        // Build the framebuffer. The image must be attached in the same order as they were defined
        // with the `ordered_passes_renderpass!` macro.
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![
                    final_image.clone(),
                    self.position_buffer.clone(),
                    self.diffuse_buffer.clone(),
                    self.normals_buffer.clone(),
                    self.depth_buffer.clone()
                ],
                ..Default::default()
            },
        )
        .unwrap();

        // Start the command buffer builder that will be filled throughout the frame handling.
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
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some(1.0f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap();

        Frame {
            system: self,
            before_main_cb_future: Some(Box::new(before_future)),
            framebuffer,
            num_pass: 0,
            command_buffer_builder: Some(command_buffer_builder),
            world_to_framebuffer,
            shadow_image,
        }
    }
}

/// Represents the active process of rendering a frame.
///
/// This struct mutably borrows the `FrameSystem`.
pub struct Frame<'a> {
    system: &'a mut FrameSystem,
    num_pass: u8,
    before_main_cb_future: Option<Box<dyn GpuFuture>>,
    framebuffer: Arc<Framebuffer>,
    command_buffer_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    world_to_framebuffer: Matrix4<f32>,
    shadow_image: Arc<AttachmentImage>,
}

impl<'a> Frame<'a> {
    /// Returns an enumeration containing the next pass of the rendering.
    pub fn next_pass<'f>(&'f mut self) -> Option<Pass<'f, 'a>> {
        // This function reads `num_pass` increments its value, and returns a struct corresponding
        // to that pass that the user will be able to manipulate in order to customize the pass.
        match {
            let current_pass = self.num_pass;
            self.num_pass += 1;
            current_pass
        } {
            0 => {
                // If we are in pass 1 then we have finished drawing the objects on the scene.
                // Going to the next subpass.
                self.command_buffer_builder
                    .as_mut()
                    .unwrap()
                    .next_subpass(SubpassContents::SecondaryCommandBuffers)
                    .unwrap();

                // And returning an object that will allow the user to apply lighting to the scene.
                Some(Pass::Lighting(LightingPass { frame: self }))
            }

            1 => {
                // If we are in pass 2 then we have finished applying lighting.
                // We take the builder, call `end_render_pass()`, and then `build()` it to obtain
                // an actual command buffer.
                self.command_buffer_builder
                    .as_mut()
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let command_buffer = self.command_buffer_builder.take().unwrap().build().unwrap();

                // Extract `before_main_cb_future` and append the command buffer execution to it.
                let after_main_cb = self
                    .before_main_cb_future
                    .take()
                    .unwrap()
                    .then_execute(self.system.gfx_queue.clone(), command_buffer)
                    .unwrap();
                // We obtain `after_main_cb`, which we give to the user.
                Some(Pass::Finished(Box::new(after_main_cb)))
            }

            // If the pass is over 2 then the frame is in the finished state and can't do anything
            // more.
            _ => None,
        }
    }
}

/// Struct provided to the user that allows them to customize or handle the pass.
pub enum Pass<'f, 's: 'f> {
    /// We are in the pass where we add lighting to the scene. The `LightingPass` allows the user
    /// to add light sources.
    Lighting(LightingPass<'f, 's>),

    /// The frame has been fully prepared, and here is the future that will perform the drawing
    /// on the image.
    Finished(Box<dyn GpuFuture>),
}

/// Allows the user to apply lighting on the scene.
pub struct LightingPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> LightingPass<'f, 's> {

    /// Applies an directional lighting to the scene.
    ///
    /// All the objects will be colored with an intensity varying between `[0, 0, 0]` and `color`,
    /// depending on the dot product of their normal and `direction`.
    pub fn directional_light(
        &mut self, direction: Vector3<f32>, 
        color: [f32; 3],
        light_view: Matrix4<f32>,
        light_projection: Matrix4<f32>,
        world: Matrix4<f32>,
        world_view_model: Matrix4<f32>,
        position_input: Arc<dyn ImageViewAbstract + 'static>,
        color_input: Arc<dyn ImageViewAbstract + 'static>,
        normals_input: Arc<dyn ImageViewAbstract + 'static>,
    ) {
        let command_buffer = self.frame.system.directional_lighting_system.draw(
            self.frame.framebuffer.extent(),
            position_input,
            color_input,
            normals_input,
            direction,
            light_view,
            light_projection,
            world,
            world_view_model,
            color,
            self.frame.shadow_image.clone()
        );
        self.frame
            .command_buffer_builder
            .as_mut()
            .unwrap()
            .execute_commands(command_buffer)
            .unwrap();
    }


    /// Returns the dimensions in pixels of the viewport.
    #[inline]
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        self.frame.framebuffer.extent()
    }
}
