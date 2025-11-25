#!/usr/bin/env python3

"""
OpenVLA Integration for Mobile ALOHA

This module integrates OpenVLA (Open Vision-Language-Action) models
with the Mobile ALOHA robot for natural language control.

OpenVLA Repository: https://github.com/openvla/openvla
Paper: https://arxiv.org/abs/2406.09246
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from PIL import Image

from aloha.vlm_controller import VLMController, VLMAction


class OpenVLAController(VLMController):
    """
    OpenVLA-based controller for Mobile ALOHA.
    
    OpenVLA is a 7B parameter vision-language-action model trained on
    large-scale robot manipulation datasets. It can understand natural
    language commands and predict robot actions.
    """
    
    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        verbose_logging: bool = True,
        **kwargs
    ):
        """
        Initialize OpenVLA controller.
        
        Args:
            model_path: HuggingFace model path or local path
            device: Device to run model on ('cuda' or 'cpu')
            load_in_8bit: Use 8-bit quantization (reduces memory)
            load_in_4bit: Use 4-bit quantization (further reduces memory)
            verbose_logging: Enable detailed input/output logging
            **kwargs: Additional arguments for VLMController
        """
        super().__init__(**kwargs)
        
        self.device = device
        self.model = None
        self.processor = None
        self.verbose_logging = verbose_logging
        self.step_counter = 0
        
        print(f"Loading OpenVLA model from {model_path}...")
        print(f"Device: {device}")
        print(f"Verbose logging: {verbose_logging}")
        
        try:
            self._load_model(model_path, load_in_8bit, load_in_4bit)
            print("OpenVLA model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load OpenVLA model: {e}")
            print("Falling back to dummy mode for testing")
            self.model = None
    
    def _load_model(self, model_path: str, load_in_8bit: bool, load_in_4bit: bool):
        """Load the OpenVLA model and processor."""
        import sys
        
        # Debug: Show Python environment info
        print(f"\nDebug Info:")
        print(f"  Python: {sys.executable}")
        print(f"  Python version: {sys.version}")
        
        # Check if transformers is available
        try:
            import transformers
            print(f"  Transformers: {transformers.__version__} ✓")
        except ImportError:
            print(f"  Transformers: NOT FOUND ✗")
            print(f"\n  Your pip shows transformers installed, but Python can't import it.")
            print(f"  This usually means you're running a different Python than where you installed.")
            print(f"  Try: {sys.executable} -m pip install transformers accelerate")
            raise ImportError("transformers not found in current Python environment")
        
        # Check for accelerate
        try:
            import accelerate
            print(f"  Accelerate: {accelerate.__version__} ✓")
        except ImportError:
            print(f"  Accelerate: NOT FOUND ✗")
            print(f"  Install with: {sys.executable} -m pip install accelerate")
        
        # Check for torch
        print(f"  PyTorch: {torch.__version__} ✓")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print()
        
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            print(f"Loading OpenVLA from {model_path}...")
            print(f"This will download the model (~14GB) on first use.")
            print(f"Please be patient, this may take several minutes...\n")
            
            # Prepare quantization config
            quantization_kwargs = {}
            if load_in_8bit:
                print("Using 8-bit quantization")
                quantization_kwargs['load_in_8bit'] = True
            elif load_in_4bit:
                print("Using 4-bit quantization")
                quantization_kwargs['load_in_4bit'] = True
            
            # Load processor
            print("Step 1/2: Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print("  ✓ Processor loaded")
            
            # Load model
            print("Step 2/2: Loading model (this is the slow part)...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **quantization_kwargs
            ).to(self.device)
            
            self.model.eval()
            print("  ✓ Model loaded and ready!")
            
        except ImportError as e:
            print(f"\n✗ Import Error: {e}")
            raise
        except Exception as e:
            print(f"\n✗ Error loading model: {e}")
            print(f"\nThis could be:")
            print(f"  1. Model not found at {model_path}")
            print(f"  2. Network issue downloading the model")
            print(f"  3. Insufficient disk space (~14GB needed)")
            print(f"  4. Insufficient VRAM (8GB+ recommended)")
            raise
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> VLMAction:
        """
        Predict action using OpenVLA model.
        
        Args:
            observation: Current observation with images and robot state
            task_description: Natural language task description
            history: Previous observations and actions
            
        Returns:
            VLMAction with predicted actions
        """
        self.step_counter += 1
        
        if self.model is None:
            # Dummy mode for testing
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[OPENVLA DUMMY MODE - Step {self.step_counter}]")
                print(f"{'='*70}")
            return self._dummy_predict(observation, task_description, history)
        
        try:
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[OPENVLA INPUT - Step {self.step_counter}]")
                print(f"{'='*70}")
            
            # Prepare image input (use main camera)
            images = observation['images']
            image = images.get('cam_high') or images.get('cam_left_wrist')
            
            if image is None:
                print("Warning: No valid image found")
                return VLMAction(reasoning="No image available")
            
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Log image info
            if self.verbose_logging:
                print(f"Image source: {'cam_high' if 'cam_high' in images else 'cam_left_wrist'}")
                print(f"Image size: {image.size}")
                print(f"Image mode: {image.mode}")
            
            # Prepare prompt
            prompt = f"Task: {task_description}"
            
            if self.verbose_logging:
                print(f"\nPrompt text: \"{prompt}\"")
                print(f"Task description: \"{task_description}\"")
                print(f"History length: {len(history)} steps")
                
                # Show robot state if available
                if 'robot_state' in observation:
                    state = observation['robot_state']
                    print(f"\nRobot State:")
                    if 'left_arm_qpos' in state:
                        print(f"  Left arm: {np.array2string(state['left_arm_qpos'], precision=3, suppress_small=True)}")
                    if 'right_arm_qpos' in state:
                        print(f"  Right arm: {np.array2string(state['right_arm_qpos'], precision=3, suppress_small=True)}")
            
            # Process inputs
            if self.verbose_logging:
                print(f"\nProcessing inputs for model...")
            
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            if self.verbose_logging:
                print(f"Input tensor shapes:")
                for key, value in inputs.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
            
            # Generate action
            if self.verbose_logging:
                print(f"\nGenerating model output...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            
            # Decode output
            action_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[OPENVLA OUTPUT - Step {self.step_counter}]")
                print(f"{'='*70}")
                print(f"Raw model output:")
                print(f"  Length: {len(action_text)} characters")
                print(f"  Text: \"{action_text}\"")
                print(f"\nOutput tokens shape: {outputs.shape}")
            
            # Parse action from model output
            action = self._parse_openvla_output(
                action_text,
                observation,
                task_description
            )
            
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[PARSED ACTION - Step {self.step_counter}]")
                print(f"{'='*70}")
                print(f"Base action: {action.base_action}")
                print(f"  Linear velocity: {action.base_action[0]:.3f} m/s")
                print(f"  Angular velocity: {action.base_action[1]:.3f} rad/s")
                if action.left_arm_action is not None:
                    print(f"Left arm action: {np.array2string(action.left_arm_action, precision=3, suppress_small=True)}")
                if action.right_arm_action is not None:
                    print(f"Right arm action: {np.array2string(action.right_arm_action, precision=3, suppress_small=True)}")
                print(f"Reasoning: \"{action.reasoning}\"")
                print(f"Confidence: {action.confidence:.2f}")
                print(f"Task done: {action.done}")
                print(f"{'='*70}\n")
            
            return action
            
        except Exception as e:
            print(f"\n{'!'*70}")
            print(f"[ERROR in OpenVLA prediction]")
            print(f"{'!'*70}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'!'*70}\n")
            return VLMAction(reasoning=f"Error: {str(e)}")
    
    def _parse_openvla_output(
        self,
        output_text: str,
        observation: Dict[str, Any],
        task_description: str,
    ) -> VLMAction:
        """
        Parse OpenVLA model output into VLMAction.
        
        OpenVLA typically outputs action sequences or descriptions.
        This method converts them to robot commands.
        """
        # Extract action from text (model-specific parsing)
        # OpenVLA outputs are typically in the form of action tokens
        # that need to be decoded to continuous actions
        
        # For now, implement a simple parser
        # In practice, you'd use the model's specific action head
        
        current_state = observation.get('robot_state', {})
        
        # Default: maintain current position with small movements
        left_arm = current_state.get('left_arm_qpos', np.zeros(6))
        right_arm = current_state.get('right_arm_qpos', np.zeros(6))
        
        # Parse movement intentions from text
        reasoning = output_text
        base_action = np.zeros(2)
        confidence = 0.8
        
        # Simple heuristic parsing (should be replaced with proper action decoder)
        text_lower = output_text.lower()
        task_lower = task_description.lower()
        
        if 'forward' in text_lower or 'move forward' in text_lower:
            base_action[0] = 0.2  # linear velocity
            reasoning = "Moving forward"
        elif 'backward' in text_lower or 'move backward' in text_lower:
            base_action[0] = -0.2
            reasoning = "Moving backward"
        elif 'left' in text_lower and 'turn' in text_lower:
            base_action[1] = 0.3  # angular velocity
            reasoning = "Turning left"
        elif 'right' in text_lower and 'turn' in text_lower:
            base_action[1] = -0.3
            reasoning = "Turning right"
        elif 'stop' in text_lower:
            base_action = np.zeros(2)
            reasoning = "Stopping"
        
        # Improved completion detection - only mark done if explicitly stated
        # AND the completion phrase directly follows task completion indicators
        done = False
        
        # Very strict completion checking - requires explicit "task complete" phrases
        completion_phrases = [
            'task complete',
            'task finished',
            'goal achieved',
            'objective complete',
            'mission accomplished'
        ]
        
        # Only mark as done if we see explicit completion phrases
        # AND it's not just describing what to do when complete
        if any(phrase in text_lower for phrase in completion_phrases):
            # Additional check: make sure it's not in a conditional context
            if not any(conditional in text_lower for conditional in ['when', 'if', 'until', 'after']):
                done = True
                reasoning = "Task objective achieved"
        
        return VLMAction(
            base_action=base_action,
            left_arm_action=left_arm,
            right_arm_action=right_arm,
            reasoning=reasoning,
            confidence=confidence,
            done=done,
        )
    
    def reason_about_task(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """
        Generate reasoning about the current state using OpenVLA.
        
        Args:
            observation: Current observation
            task_description: Task description
            history: Execution history
            
        Returns:
            Reasoning text
        """
        if self.model is None:
            return "Model not loaded - using dummy reasoning"
        
        try:
            # Prepare image
            images = observation['images']
            image = images.get('cam_high') or images.get('cam_left_wrist')
            
            if image is None:
                return "No image available for reasoning"
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Create reasoning prompt
            prompt = f"Observe the scene and explain what you see. Task: {task_description}"
            
            # Process
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate reasoning
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                )
            
            reasoning = self.processor.decode(outputs[0], skip_special_tokens=True)
            return reasoning
            
        except Exception as e:
            return f"Error generating reasoning: {str(e)}"
    
    def _dummy_predict(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> VLMAction:
        """
        Dummy prediction for testing without model.
        
        Implements simple rule-based behavior for testing the system.
        Now task-aware and more flexible.
        """
        step = len(history)
        task_lower = task_description.lower()
        
        # Parse task to determine appropriate behavior
        # More intelligent state machine based on task description
        base_action = np.array([0.0, 0.0])
        reasoning = ""
        done = False
        
        # Task-specific behavior
        if 'explore' in task_lower:
            # Exploration pattern - continues longer
            cycle = step % 30
            if cycle < 8:
                base_action = np.array([0.2, 0.0])
                reasoning = f"Exploring: moving forward (step {step+1})"
            elif cycle < 16:
                base_action = np.array([0.0, 0.4])
                reasoning = f"Exploring: rotating to scan (step {step+1})"
            elif cycle < 24:
                base_action = np.array([0.2, 0.0])
                reasoning = f"Exploring: moving to new area (step {step+1})"
            else:
                base_action = np.array([0.0, -0.4])
                reasoning = f"Exploring: rotating back (step {step+1})"
            
        elif 'forward' in task_lower:
            # Simple forward movement
            if step < 20:
                base_action = np.array([0.2, 0.0])
                reasoning = f"Moving forward as requested (step {step+1})"
            else:
                base_action = np.array([0.0, 0.0])
                reasoning = "Forward movement sufficient"
                done = True
                
        elif 'turn' in task_lower or 'rotate' in task_lower:
            # Rotation behavior
            direction = 1.0 if 'left' in task_lower else -1.0
            if step < 15:
                base_action = np.array([0.0, direction * 0.3])
                reasoning = f"Rotating {'left' if direction > 0 else 'right'} (step {step+1})"
            else:
                base_action = np.array([0.0, 0.0])
                reasoning = "Rotation complete"
                done = True
                
        elif 'navigate' in task_lower or 'go to' in task_lower:
            # Navigation pattern
            if step < 10:
                base_action = np.array([0.2, 0.0])
                reasoning = f"Navigating: moving toward target (step {step+1})"
            elif step < 15:
                base_action = np.array([0.0, 0.3])
                reasoning = f"Navigating: adjusting orientation (step {step+1})"
            elif step < 25:
                base_action = np.array([0.2, 0.0])
                reasoning = f"Navigating: approaching target (step {step+1})"
            else:
                base_action = np.array([0.0, 0.0])
                reasoning = "Navigation target reached"
                done = True
        else:
            # Default exploration behavior for unknown tasks
            if step < 10:
                base_action = np.array([0.2, 0.0])
                reasoning = f"Executing task: moving forward (step {step+1})"
            elif step < 20:
                base_action = np.array([0.0, 0.3])
                reasoning = f"Executing task: scanning area (step {step+1})"
            elif step < 30:
                base_action = np.array([0.15, 0.0])
                reasoning = f"Executing task: continuing motion (step {step+1})"
            else:
                base_action = np.array([0.0, 0.0])
                reasoning = "Task execution complete"
                done = True
        
        # Get current arm positions
        current_state = observation.get('robot_state', {})
        left_arm = current_state.get('left_arm_qpos', np.zeros(6))
        right_arm = current_state.get('right_arm_qpos', np.zeros(6))
        
        return VLMAction(
            base_action=base_action,
            left_arm_action=left_arm,
            right_arm_action=right_arm,
            reasoning=reasoning,
            confidence=0.6,
            done=done,
        )


class OpenVLAFineTunedController(OpenVLAController):
    """
    OpenVLA controller with Mobile ALOHA-specific fine-tuning.
    
    This variant loads a version of OpenVLA that has been fine-tuned
    on Mobile ALOHA datasets for better performance on manipulation tasks.
    """
    
    def __init__(
        self,
        model_path: str = "openvla/openvla-7b-mobile-aloha",
        **kwargs
    ):
        """
        Initialize fine-tuned OpenVLA controller.
        
        Args:
            model_path: Path to fine-tuned model
            **kwargs: Additional arguments
        """
        print("Initializing Mobile ALOHA fine-tuned OpenVLA model")
        super().__init__(model_path=model_path, **kwargs)
    
    def _parse_openvla_output(
        self,
        output_text: str,
        observation: Dict[str, Any],
        task_description: str,
    ) -> VLMAction:
        """
        Parse output from fine-tuned model.
        
        Fine-tuned models may output in a different format optimized
        for Mobile ALOHA's action space.
        """
        # This would be customized based on how the model was fine-tuned
        # For now, use the base implementation
        return super()._parse_openvla_output(output_text, observation, task_description)
