use std::{sync::Arc,};

use cgmath::{Matrix4, Rad, Vector3, Point3, SquareMatrix};
use vulkano::sync::GpuFuture;
use crate::{scene_pkg::scene::Scene};

use super::{object_3d_shadow_pass::Object3DShadowPass, shadow_map_renderer::ShadowMapRenderer};

pub struct SceneShadowPass {
    object_3d_passes: Vec<Object3DShadowPass>,
    scene: Arc<Scene>
}

impl SceneShadowPass {
    // public methods
    pub fn new(scene: Arc<Scene>, shadow_map_renderer: &ShadowMapRenderer) -> SceneShadowPass {

        let mut object_3d_passes: Vec<Object3DShadowPass> = vec![];
        object_3d_passes.reserve(scene.objects.len());
        for object_3d in scene.objects.clone()  {
            object_3d_passes.push(Object3DShadowPass::new(
                &shadow_map_renderer,
                object_3d
            ));
        }


        SceneShadowPass {
            object_3d_passes,
            scene
        }
    }

    pub fn draw(&mut self, shadow_map_renderer: &mut ShadowMapRenderer) {

        let (view, projection) = self.scene.directional_lights[0].clone().view_projection();
        let world = self.scene.world_model;

        for i in 0..self.object_3d_passes.len() {
            let cb = self.object_3d_passes[i].draw(shadow_map_renderer, world, view, projection);
            shadow_map_renderer.execute_draw_pass(cb);
        }


    }


    // private methods
    

}