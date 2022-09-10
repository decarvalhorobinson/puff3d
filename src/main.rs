
mod config;
mod system;
mod frame;
mod mesh_converters;


use crate::{config::vulkan::vulkan_init, mesh_converters::ObjFileToMeshConverter};

fn main() {
    println!("I am Puff3D a small engine made for a passionate newbiew rust developer!");
    vulkan_init();

}
