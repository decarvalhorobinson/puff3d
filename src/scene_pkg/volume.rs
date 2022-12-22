use cgmath::{Matrix4, Rad, SquareMatrix, Vector3, Deg};

use super::{material::Material, mesh::Mesh, transform::Transform};

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Volume {
    pub transform: Transform,
    pub pixel_data: Vec<u16>,
    pub dimension: [u32; 3],
    pub model_matrix: Matrix4<f32>,
}

impl Volume {
    pub fn new() -> Volume {
        let transform = Transform {
            position: Vector3::new(0.0f32, 0.0f32, 0.0f32),
            scale: Vector3::new(1.0f32, 1.00f32, 1.0f32),
            rotation: Vector3::new(0.0f32, 0.0f32, 0.0f32),
        };
        let mut volume = Volume {
            transform: transform,
            pixel_data: vec![],
            dimension: [0, 0, 0],
            model_matrix: Matrix4::identity()
        };
        Volume::update_model_matrix(&mut volume);

        volume
    }

    pub fn update_model_matrix(&mut self) {
        let mut model = Matrix4::identity();
        
        model = model
            * Matrix4::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), Deg(self.transform.rotation.x));
        model = model
            * Matrix4::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Deg(self.transform.rotation.y));
        model = model
            * Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Deg(self.transform.rotation.z));
        model = model * Matrix4::from_translation(self.transform.position);
        model = model
            * Matrix4::from_nonuniform_scale(
                self.transform.scale.x,
                self.transform.scale.y,
                self.transform.scale.z,
            );

        self.model_matrix = model;
    }
}
