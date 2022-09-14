
mod config;
mod system;
mod frame;
mod object_3d_loader;
mod scene_pkg;


use crate::{config::vulkan::vulkan_init};

fn main() {
    println!("I am Puff3D a small engine made for a passionate newbiew rust developer!");
    vulkan_init();

}
