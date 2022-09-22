use std::ops;

use bytemuck::{Pod, Zeroable};
use vulkano::impl_vertex;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}

impl_vertex!(Vertex, position);

impl ops::Sub<Vertex> for Vertex {
    type Output = Vertex;

    fn sub(self, rhs: Vertex) -> Self::Output {
        Vertex {
            position: [
                self.position[0] - rhs.position[0],
                self.position[1] - rhs.position[1],
                self.position[2] - rhs.position[2],
            ],
        }
    }
}

impl ops::Mul<f32> for Vertex {
    type Output = Vertex;

    fn mul(self, rhs: f32) -> Self::Output {
        Vertex {
            position: [
                self.position[0] * rhs,
                self.position[1] * rhs,
                self.position[2] * rhs,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    pub normal: [f32; 3],
}

impl_vertex!(Normal, normal);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Tangent {
    pub tangent: [f32; 3],
}

impl_vertex!(Tangent, tangent);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Uv {
    pub uv: [f32; 2],
}

impl_vertex!(Uv, uv);

impl ops::Sub<Uv> for Uv {
    type Output = Uv;

    fn sub(self, rhs: Uv) -> Self::Output {
        Uv {
            uv: [self.uv[0] - rhs.uv[0], self.uv[1] - rhs.uv[1]],
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
    pub uvs: Vec<Uv>,
    pub tangent: Vec<Tangent>,
    pub indices: Vec<u16>,
}
