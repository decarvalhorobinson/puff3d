use cgmath::{Matrix4, Rad, SquareMatrix, Vector3};

use super::{material::Material, mesh::Mesh, transform::Transform};

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Object3D {
    pub transform: Transform,
    pub mesh: Mesh,
    pub material: Material,
    pub visibility: bool,
    pub model_matrix: Matrix4<f32>,
}

impl Object3D {
    pub fn new() -> Object3D {
        let transform = Transform {
            position: Vector3::new(0.0f32, 0.0f32, 0.0f32),
            scale: Vector3::new(1.0f32, 1.0f32, 1.0f32),
            rotation: Vector3::new(0.0f32, 0.0f32, 0.0f32),
        };
        let mut obj = Object3D {
            transform: transform,
            mesh: Mesh {
                vertices: vec![],
                normals: vec![],
                uvs: vec![],
                tangent: vec![],
                indices: vec![],
            },
            material: Material {
                diffuse_file_path: String::new(),
                normal_file_path: String::new(),
                metallic_file_path: String::new(),
                roughness_file_path: String::new(),
                ao_file_path: String::new()
            },
            visibility: true,
            model_matrix: Matrix4::identity(),
        };
        Object3D::update_model_matrix(&mut obj);

        obj
    }

    pub fn update_model_matrix(&mut self) {
        let mut model = Matrix4::identity();
        model = model * Matrix4::from_translation(self.transform.position);
        model = model
            * Matrix4::from_nonuniform_scale(
                self.transform.scale.x,
                self.transform.scale.y,
                self.transform.scale.z,
            );
        model = model
            * Matrix4::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), Rad(self.transform.rotation.x));
        model = model
            * Matrix4::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(self.transform.rotation.y));
        model = model
            * Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Rad(self.transform.rotation.z));

        self.model_matrix = model;
    }
}
