use std::{sync::Arc, rc::Rc};

use cgmath::{Matrix4, Rad, Vector3, Point3, SquareMatrix};
use vulkano::render_pass::Subpass;
use crate::{scene_pkg::scene::Scene, frame::DrawPass};
use super::{draw_system::DrawSystem, object_3d_draw_system::Object3DDrawSystem};


pub struct SceneDrawSystem {
    obj_draw_systems: Vec<Object3DDrawSystem>,
    scene: Arc<Scene>
}

impl SceneDrawSystem {
    // public methods
    pub fn new(scene: Arc<Scene>, draw_system: Arc<DrawSystem>) -> SceneDrawSystem {

        let mut obj_draw_systems: Vec<Object3DDrawSystem> = vec![];
        obj_draw_systems.reserve(scene.objects.len());
        for object_3d in scene.objects.clone()  {
            obj_draw_systems.push(Object3DDrawSystem::new(
                draw_system.clone(),
                object_3d
            ));
        }


        SceneDrawSystem {
            scene: scene.clone(),
            obj_draw_systems: obj_draw_systems
        }
    }

    pub fn draw_deferred(&mut self, mut draw_pass: DrawPass) {        
        let (world, view, projection) = self.get_camera_matrices(draw_pass.viewport_dimensions());

        for i in 0..self.obj_draw_systems.len() {
            let mut obj_draw_systems = self.obj_draw_systems[i].clone();
            let cb = obj_draw_systems.draw_deferred(draw_pass.viewport_dimensions(), world, projection, view);
            draw_pass.execute(cb);
        }


    }

    pub fn get_camera_matrices(&mut self, viewport_dimensions: [u32; 2]) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
        (self.scene.world_model, self.scene.active_camera.get_view_matrix(), Scene::projection(viewport_dimensions))

    }

    // private methods
    

}