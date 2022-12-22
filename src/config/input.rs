use std::collections::HashSet;

use winit::event::{DeviceEvent, ElementState, Event, VirtualKeyCode};

#[derive(Debug)]
pub struct Input {
    pub key_pressed: HashSet<VirtualKeyCode>,
    pub key_held: HashSet<VirtualKeyCode>,
    pub key_released: HashSet<VirtualKeyCode>,

    pub mouse_dx: f32,
    pub mouse_dy: f32,

    pub clear_mouse: bool
}

impl Input {
    pub fn new() -> Input {
        Input {
            key_pressed: HashSet::new(),
            key_held: HashSet::new(),
            key_released: HashSet::new(),

            mouse_dx: 0.0,
            mouse_dy: 0.0,

            clear_mouse: false
        }
    }

    pub fn update(&mut self, event: &Event<()>) -> bool {
        let mut redraw: bool = false;

        self.key_held.extend(self.key_pressed.clone());
        self.key_pressed.clear();
        
        if self.clear_mouse {
            self.handle_mouse_motion(0.0, 0.0);
        }

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                self.handle_mouse_motion(delta.0 as f32, delta.1 as f32);
                self.clear_mouse = false;
            }
            Event::DeviceEvent {
                event: DeviceEvent::Key(input),
                ..
            } => {
                self.handle_keyboard_input(input.virtual_keycode.unwrap_or_else(|| VirtualKeyCode::F19), input.state);
            }
            Event::RedrawEventsCleared => {
                redraw = true;
                self.clear_mouse = true;
            }
            _ => {  }
        }

        redraw
    }

    pub fn handle_keyboard_input(&mut self, key: VirtualKeyCode, state: ElementState) {
        if state == ElementState::Pressed {
            self.key_pressed.insert(key);
            return;
        }

        if state == ElementState::Released {
            self.key_released.insert(key);
            self.key_held.remove(&key);
            return;
        }
    }

    pub fn key_pressed(&self, key: VirtualKeyCode) -> bool {
        self.key_pressed.contains(&key)
    }

    pub fn key_held(&self, key: VirtualKeyCode) -> bool {
        self.key_held.contains(&key)
    }

    pub fn key_released(&self, key: VirtualKeyCode) -> bool {
        self.key_released.contains(&key)
    }

    pub fn handle_mouse_motion(&mut self, dx: f32, dy: f32) {
        (self.mouse_dx, self.mouse_dy) = (dx, dy);
    }
}
