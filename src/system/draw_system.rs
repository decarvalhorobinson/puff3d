use std::sync::Arc;

use vulkano::{device::Queue, render_pass::Subpass};

#[repr(C)]
#[derive(Clone, Debug)]
pub struct DrawSystem {

    pub gfx_queue: Arc<Queue>,
    pub render_pass: Arc<vulkano::render_pass::RenderPass>,

}