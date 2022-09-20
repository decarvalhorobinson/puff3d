use std::{sync::{Arc, Mutex}, rc::Rc};

use cgmath::{Matrix4, Rad, Vector3, Point3, SquareMatrix};
use vulkano::render_pass::Subpass;
use crate::{scene_pkg::scene::Scene, frame::DrawPass};
use super::{draw_system::DrawSystem, object_3d_draw_system::Object3DDrawSystem};


pub struct SceneDrawSystem {
    obj_draw_systems: Vec<Object3DDrawSystem>,
    scene: Arc<Mutex<Scene>>,
    rotation: f32
}

impl SceneDrawSystem {
    // public methods
    pub fn new(scene: Arc<Mutex<Scene>>, draw_system: Arc<DrawSystem>) -> SceneDrawSystem {

        let scene_locked = scene.lock().unwrap();
        let mut obj_draw_systems: Vec<Object3DDrawSystem> = vec![];
        obj_draw_systems.reserve(scene_locked.objects.len());
        for object_3d in scene_locked.objects.clone()  {
            obj_draw_systems.push(Object3DDrawSystem::new(
                draw_system.clone(),
                object_3d
            ));
        }


        SceneDrawSystem {
            scene: scene.clone(),
            obj_draw_systems: obj_draw_systems,
            rotation: 0.0
        }
    }

    pub fn draw_deferred(&mut self, mut draw_pass: DrawPass) {        
        let (mut world, view, projection) = self.get_camera_matrices(draw_pass.viewport_dimensions());

        for i in 0..self.obj_draw_systems.len() {
            let mut obj_draw_systems = self.obj_draw_systems[i].clone();
            let cb = obj_draw_systems.draw_deferred(draw_pass.viewport_dimensions(), world, projection, view);
            draw_pass.execute(cb);
        }


    }

    pub fn get_camera_matrices(&mut self, viewport_dimensions: [u32; 2]) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
        let scene_locked = self.scene.lock().unwrap();
        (scene_locked.world_model, scene_locked.active_camera.get_view_matrix(), Scene::projection(viewport_dimensions))

    }

    // private methods
    

}