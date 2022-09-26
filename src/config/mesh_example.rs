use std::sync::{Arc, Mutex};

use cgmath::{Matrix4, Point3, SquareMatrix, Vector3};

use crate::{
    object_3d_loader::mesh_converters::ObjFileToMeshConverter,
    scene_pkg::{
        camera::Camera, directional_light::DirectionalLight, mesh::Mesh, object3d::Object3D,
        scene::Scene,
    },
};

pub fn get_example_mesh_cottage_house() -> Mesh {
    let obj_to_mesh_converter =
        ObjFileToMeshConverter::new(String::from("./src/cottage_house/cottage.obj"));
    obj_to_mesh_converter.create_mesh()
}

pub fn get_example_scene_cottage_house() -> std::sync::Arc<Mutex<Scene>> {
    let mut scene = Scene {
        cameras: vec![],
        active_camera: Camera::new(),
        objects: vec![],
        world_model: Matrix4::identity(),
        directional_lights: vec![Arc::new(Mutex::new(DirectionalLight {
            position: Point3::new(-40.0, 30.0, -40.0),
            center: Point3::new(0.0, 0.0, 0.0),
            color: [1.0, 1.0, 1.0, 1.0],
        }))],
    };

    let mesh = get_example_mesh_cottage_house();

    let obj_to_mesh_converter =
        ObjFileToMeshConverter::new(String::from("./src/brick_wall/brick_wall.obj"));
    let mut obj = Object3D::new();
    obj.mesh = obj_to_mesh_converter.create_mesh();
    obj.material.diffuse_file_path = "./src/brick_wall/brickwall.png".into();
    obj.material.normal_file_path = "./src/brick_wall/brickwall_normal.png".into();

    obj.update_model_matrix();
    scene.objects.push(obj);

    /*let mut obj = Object3D::new();
    obj.mesh = get_example_mesh_cottage_house();
    obj.material.diffuse_file_path = "./src/cottage_house/cottage_diffuse.png".into();
    obj.material.normal_file_path = "./src/cottage_house/cottage_normal.png".into();
    obj.update_model_matrix();
    scene.objects.push(obj);*/


    let obj_to_mesh_converter =
        ObjFileToMeshConverter::new(String::from("./src/watch_tower/watch_tower.obj"));
    let mesh = obj_to_mesh_converter.create_mesh();
    let mut obj = Object3D::new();
    obj.mesh = mesh;
    obj.material.diffuse_file_path = "./src/watch_tower/textures/diffuse.png".into();
    obj.material.normal_file_path = "./src/watch_tower/textures/normals.png".into();
    obj.update_model_matrix();
    scene.objects.push(obj);

    std::sync::Arc::new(Mutex::new(scene))
}
