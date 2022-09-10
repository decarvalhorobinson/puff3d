use bytemuck::{Pod, Zeroable};
use vulkano::impl_vertex;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}

impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    pub normal: [f32; 3],
}

impl_vertex!(Normal, normal);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Uv {
    pub uv: [f32; 2],
}

impl_vertex!(Uv, uv);

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
    pub uvs: Vec<Uv>,
    pub indices: Vec<u16>
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, normals: Vec<Normal>, uvs: Vec<Uv>, indices: Vec<u16>) -> Mesh {
        Mesh {
            vertices,
            normals,
            uvs,
            indices
        }
    }
}
