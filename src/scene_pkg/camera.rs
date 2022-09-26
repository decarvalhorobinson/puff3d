use cgmath::{Angle, EuclideanSpace, Euler, InnerSpace, Matrix4, Point3, Rad, Vector3, Deg};

#[derive(PartialEq)]
pub enum CameraMovement {
    Forward,
    Backward,
    Left,
    Right,
    Top,
    Down,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Camera {
    // Camera position vec3(x, y, z)
    pub position: Point3<f32>,
    // Camera target vec3(x, y, z)
    pub target: Point3<f32>,
    // Camera vector to right direction related to the camera view vec3(x, y, z)
    pub right_direction: Vector3<f32>,
    // World´s vector to up direction related to the camera vec3(x, y, z)
    pub up_direction: Vector3<f32>,
    // World´s vector to up direction related to the world vec3(x, y, z)
    pub world_up_direction: Vector3<f32>,
    // Camera rotate vec3(pitch, yaw, roll) pitch = x axis, yaw = y axis and roll = z axis
    pub pitch_yaw_roll: Vector3<f32>,
    //last position of the camera related to the update
    pub last_position: Point3<f32>,
    // field of view, camera angle of vision
    pub field_of_view: f32,
    //  how close a vetice can be drawn to the camera
    pub z_near: f32,
    // how far a vetice can be drawn to the camera
    pub z_far: f32,
    // how far a vetice can be drawn to the camera
    pub mouse_sensitivity: f32,
    // how far a vetice can be drawn to the camera
    pub movement_speed: f32,
}

const SPEED: f32 = 0.05f32;
const SENSITIVTY: f32 = 0.5f32;
const ZOOM: f32 = 72.0f32;

impl Camera {
    pub fn new() -> Camera {
        let position = Point3::new(-40.0, 20.0, 20.0);
        let last_position = Point3::new(-10.0, 10.0, 0.0);
        let world_up_direction = Vector3::new(0.0, -1.0, 0.0);
        let pitch_yaw_roll = Vector3::new(0.0, 0.0, 0.0);
        let target = Point3::new(0.0, 0.0, -1.0);
        let movement_speed = SPEED;
        let mouse_sensitivity = SENSITIVTY;
        let field_of_view = ZOOM;
        let z_near = 0.05;
        let z_far = 100.0;

        let mut camera = Camera {
            position: position,
            target: target,
            right_direction: Vector3::new(0.0f32, 0.0f32, 0.0f32),
            up_direction: Vector3::new(0.0f32, -1.0f32, 0.0f32),
            world_up_direction: world_up_direction,
            pitch_yaw_roll: pitch_yaw_roll,
            last_position: last_position,
            field_of_view: field_of_view,
            z_near: z_near,
            z_far: z_far,
            mouse_sensitivity: mouse_sensitivity,
            movement_speed: movement_speed,
        };
        camera.update_camera_vectors();

        return camera;
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(
            self.position,
            self.position + self.target.to_vec(),
            self.up_direction,
        );
        let scale = Matrix4::from_scale(1.0);
        view * scale
    }

    pub fn process_mouse_movement(&mut self, mut x_offset: f32, mut y_offset: f32) {
        x_offset *= self.mouse_sensitivity;
        y_offset *= self.mouse_sensitivity;

        println!("pitch: {:?}, yaw: {:?}",  self.pitch_yaw_roll.x, self.pitch_yaw_roll.y);

        //set pitch
        self.pitch_yaw_roll.x -= y_offset;
        //set Yaw
        self.pitch_yaw_roll.y -= x_offset;

        if self.pitch_yaw_roll.x > 89.0 {
            self.pitch_yaw_roll.x = 89.0;
        }

        if self.pitch_yaw_roll.x < -89.0 {
            self.pitch_yaw_roll.x = -89.0;
        }

        // Update Front, Right and Up Vectors using the updated Eular angles
        self.update_camera_vectors();
    }

    pub fn update_camera_vectors(&mut self) {
        // Calculate the new Front vector
        let mut front = Vector3::new(0.0, 0.0, 0.0);
        front.x = Angle::cos(Deg(self.pitch_yaw_roll.y)) * Angle::cos(Deg(self.pitch_yaw_roll.x));
        front.y = Angle::sin(Deg(self.pitch_yaw_roll.x));
        front.z = Angle::sin(Deg(self.pitch_yaw_roll.y)) * Angle::cos(Deg(self.pitch_yaw_roll.x));
        self.target = Point3::from_vec(front.normalize());
        // Also re-calculate the Right and Up vector
        self.right_direction =
            Vector3::cross(self.target.to_vec(), self.world_up_direction).normalize(); // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        self.up_direction = Vector3::cross(self.right_direction, self.target.to_vec()).normalize();
    }

    pub fn process_keyboard(&mut self, direction: CameraMovement, delta_time: f32) {
        let velocity: f32 = self.movement_speed * delta_time;
        if direction == CameraMovement::Forward {
            self.position += self.target.to_vec() * velocity;
        }
        if direction == CameraMovement::Backward {
            self.position -= self.target.to_vec() * velocity;
        }
        if direction == CameraMovement::Left {
            self.position -= self.right_direction * velocity;
        }
        if direction == CameraMovement::Right {
            self.position += self.right_direction * velocity;
        }
        if direction == CameraMovement::Top {
            self.position -= self.world_up_direction * velocity;
        }
        if direction == CameraMovement::Down {
            self.position += self.world_up_direction * velocity;
        }
    }
}
