// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::{Vector3, Matrix4, SquareMatrix, Point3};

use crate::{scene_pkg::{mesh::{Vertex, Mesh, Normal}, scene::Scene, camera::Camera, directional_ligh::DirectionalLight, object3d::Object3D}, object_3d_loader::mesh_converters::ObjFileToMeshConverter};

pub fn get_example_mesh_cottage_house() -> Mesh {
    let obj_to_mesh_converter = ObjFileToMeshConverter::new(String::from("./src/cottage_house/cottage.obj"));
    obj_to_mesh_converter.create_mesh()
}

pub fn get_example_scene_cottage_house() -> std::sync::Arc<Scene> {
    
    let mut scene = Scene {
        cameras: vec![],
        active_camera: Camera::new(),
        objects: vec![],
        world_model: Matrix4::identity(),
        directional_lights: vec![
            DirectionalLight {
                position: Point3::new(-5.0, 5.0, -5.0),
                center: Point3::new(0.0, 0.0, 0.0), 
                color: [1.0, 1.0, 1.0],
            }
        ]

    };

    let obj_to_mesh_converter = ObjFileToMeshConverter::new(String::from("./src/brick_wall/brick_wall.obj"));
    let mut obj = Object3D::new();
    obj.mesh = obj_to_mesh_converter.create_mesh();
    obj.material.diffuse_file_path = "./src/brick_wall/brickwall.png".into();
    obj.material.normal_file_path = "./src/brick_wall/brickwall_normal.png".into();
    scene.objects.push(obj);

    let mut obj = Object3D::new();
    obj.mesh = get_example_mesh_cottage_house();
    obj.material.diffuse_file_path = "./src/cottage_house/cottage_diffuse.png".into();
    obj.material.normal_file_path = "./src/cottage_house/cottage_normal.png".into();
    obj.update_model_matrix();
    scene.objects.push(obj);

    std::sync::Arc::new(scene)
}

pub fn get_example_scene_brick_wall() -> Scene {

    let mut scene = Scene {
        cameras: vec![],
        active_camera: Camera::new(),
        objects: vec![],
        world_model: Matrix4::identity(),
        directional_lights: vec![
            DirectionalLight {
                position: Point3::new(5.0, 5.0, -2.0),
                center: Point3::new(0.0, 0.0, 0.0),  
                color: [1.0, 1.0, 1.0],
            }
        ]

    };

    let obj_to_mesh_converter = ObjFileToMeshConverter::new(String::from("./src/brick_wall/brick_wall.obj"));
    

    let mut obj = Object3D::new();
    obj.mesh = obj_to_mesh_converter.create_mesh();
    obj.material.diffuse_file_path = "./src/brick_wall/brickwall.png".into();
    obj.material.normal_file_path = "./src/brick_wall/brickwall_normal.png".into();

    scene.objects.push(obj);

    scene
}