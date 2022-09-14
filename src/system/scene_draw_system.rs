use std::{sync::Arc, rc::Rc};

use cgmath::{Matrix4, Rad, Vector3};
use vulkano::render_pass::Subpass;
use crate::{scene_pkg::scene::Scene, frame::DrawPass};
use super::{draw_system::DrawSystem, object_3d_draw_system::Object3DDrawSystem};


pub struct SceneDrawSystem {
    rotation_test: f32,
    mesh_draw_systems: Vec<Object3DDrawSystem>,
    scene: Scene,
    draw_system: Arc<DrawSystem>
}

impl SceneDrawSystem {
    // public methods
    pub fn new(scene: Scene, draw_system: Arc<DrawSystem>) -> SceneDrawSystem {

        let mut mesh_draw_systems: Vec<Object3DDrawSystem> = vec![];
        mesh_draw_systems.reserve(scene.objects.len());
        for object_3d in scene.objects.clone()  {
            mesh_draw_systems.push(Object3DDrawSystem::new(
                draw_system.gfx_queue.clone(), 
                Subpass::from(draw_system.render_pass.clone(), 0).unwrap(),
                object_3d
            ));
        }


        SceneDrawSystem {
            rotation_test: 0.0f32,
            scene: scene.clone(),
            draw_system: draw_system.clone(),
            mesh_draw_systems: mesh_draw_systems
        }
    }

    pub fn draw(&mut self, mut draw_pass: DrawPass) {
        self.rotation_test += 0.01;
        let world = Matrix4::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(self.rotation_test as f32));
        let viewport_dimensions = draw_pass.viewport_dimensions();
        let aspect_ratio = viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32;
        let projection = cgmath::perspective(
            Rad(std::f32::consts::FRAC_PI_2),
            aspect_ratio,
            0.01,
            100.0,
        );

        for i in 0..self.mesh_draw_systems.len() {
            let mut mesh_draw_system = self.mesh_draw_systems[i].clone();
            let cb = mesh_draw_system.draw(viewport_dimensions, world, 10.0, 0.05);
            draw_pass.execute(cb);
        }


    }
    // private methods
    

}