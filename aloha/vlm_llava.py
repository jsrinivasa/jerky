#!/usr/bin/env python3

"""
LLaVA Integration for Mobile ALOHA

LLaVA (Large Language and Vision Assistant) is a lightweight, open-source
vision-language model that runs locally without API keys.

This is a practical alternative to OpenVLA for local inference.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from PIL import Image

from aloha.vlm_controller import VLMController, VLMAction


class LLaVAController(VLMController):
    """
    LLaVA-based controller for Mobile ALOHA.
    
    LLaVA is a 7B/13B parameter model that combines vision and language
    understanding. It's well-supported, actively maintained, and runs
    efficiently on consumer GPUs.
    """
    
    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        verbose_logging: bool = True,
        **kwargs
    ):
        """
        Initialize LLaVA controller.
        
        Args:
            model_path: HuggingFace model path (default: llava-1.5-7b-hf)
            device: Device to run model on
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization (recommended)
            verbose_logging: Enable detailed logging
            **kwargs: Additional arguments for VLMController
        """
        super().__init__(**kwargs)
        
        self.device = device
        self.model = None
        self.processor = None
        self.verbose_logging = verbose_logging
        self.step_counter = 0
        
        print(f"Loading LLaVA model from {model_path}...")
        print(f"Device: {device}")
        print(f"4-bit quantization: {load_in_4bit}")
        print(f"8-bit quantization: {load_in_8bit}")
        
        try:
            self._load_model(model_path, load_in_8bit, load_in_4bit)
            print("✓ LLaVA model loaded successfully!")
            print(f"  Model: {model_path}")
            print(f"  Ready for local inference (no API keys needed)")
        except Exception as e:
            print(f"✗ Could not load LLaVA model: {e}")
            print("  Falling back to dummy mode")
            print("\nTo install LLaVA dependencies:")
            print("  pip install transformers accelerate bitsandbytes")
            self.model = None
    
    def _load_model(self, model_path: str, load_in_8bit: bool, load_in_4bit: bool):
        """Load the LLaVA model and processor."""
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        from transformers import BitsAndBytesConfig
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Prepare quantization config
        quantization_config = None
        if load_in_4bit:
            print("Configuring 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            print("Configuring 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model
        print("Loading model (this may take a minute)...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        if self.device == "cpu" or quantization_config is None:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded and ready!")
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> VLMAction:
        """
        Predict action using LLaVA model.
        
        Args:
            observation: Current observation
            task_description: Natural language task
            history: Execution history
            
        Returns:
            VLMAction with predicted actions
        """
        self.step_counter += 1
        
        if self.model is None:
            # Dummy mode
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[LLAVA DUMMY MODE - Step {self.step_counter}]")
                print(f"{'='*70}")
            return self._dummy_predict(observation, task_description, history)
        
        try:
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[LLAVA INPUT - Step {self.step_counter}]")
                print(f"{'='*70}")
            
            # Get image
            images = observation['images']
            image = images.get('cam_high') or images.get('cam_left_wrist')
            
            if image is None:
                print("Warning: No valid image found")
                return VLMAction(reasoning="No image available")
            
            # Convert to PIL
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if self.verbose_logging:
                print(f"Image source: {'cam_high' if 'cam_high' in images else 'cam_left_wrist'}")
                print(f"Image size: {image.size}")
            
            # Create detailed prompt for robot control
            prompt = self._create_robot_control_prompt(
                task_description,
                observation,
                history
            )
            
            if self.verbose_logging:
                print(f"\nPrompt: \"{prompt[:200]}...\"")
                print(f"Full prompt length: {len(prompt)} chars")
            
            # Format for LLaVA
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            prompt_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            
            # Process inputs
            if self.verbose_logging:
                print("\nProcessing inputs...")
            
            inputs = self.processor(
                images=image,
                text=prompt_text,
                return_tensors="pt"
            ).to(self.device)
            
            if self.verbose_logging:
                print("Generating response...")
            
            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                )
            
            # Decode
            generated_text = self.processor.decode(
                output[0],
                skip_special_tokens=True
            )
            
            # Extract the assistant's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text
            
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[LLAVA OUTPUT - Step {self.step_counter}]")
                print(f"{'='*70}")
                print(f"Raw output length: {len(generated_text)} chars")
                print(f"Response: \"{response}\"")
            
            # Parse into action
            action = self._parse_llava_output(
                response,
                observation,
                task_description
            )
            
            if self.verbose_logging:
                print(f"\n{'='*70}")
                print(f"[PARSED ACTION - Step {self.step_counter}]")
                print(f"{'='*70}")
                print(f"Base: linear={action.base_action[0]:.3f}, angular={action.base_action[1]:.3f}")
                print(f"Reasoning: \"{action.reasoning}\"")
                print(f"Confidence: {action.confidence:.2f}")
                print(f"Done: {action.done}")
                print(f"{'='*70}\n")
            
            return action
            
        except Exception as e:
            print(f"\n{'!'*70}")
            print(f"[ERROR in LLaVA prediction]")
            print(f"{'!'*70}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'!'*70}\n")
            return VLMAction(reasoning=f"Error: {str(e)}")
    
    def _create_robot_control_prompt(
        self,
        task_description: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:
        """Create a prompt for robot control."""
        step = len(history)
        
        prompt = f"""You are controlling a mobile robot with this task: {task_description}

Current step: {step + 1}

Analyze the image and decide the next action. Respond with:
1. What you see
2. Next action: FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, or STOP
3. Reasoning for this action

Keep response concise."""
        
        return prompt
    
    def _parse_llava_output(
        self,
        output_text: str,
        observation: Dict[str, Any],
        task_description: str,
    ) -> VLMAction:
        """Parse LLaVA output into robot action."""
        text_lower = output_text.lower()
        
        # Default values
        base_action = np.zeros(2)
        reasoning = output_text[:200]
        confidence = 0.7
        
        # Parse action from text
        if 'forward' in text_lower:
            base_action[0] = 0.2
            reasoning = "Moving forward based on visual analysis"
        elif 'backward' in text_lower or 'back' in text_lower:
            base_action[0] = -0.2
            reasoning = "Moving backward"
        elif 'turn left' in text_lower or 'left' in text_lower:
            base_action[1] = 0.3
            reasoning = "Turning left"
        elif 'turn right' in text_lower or 'right' in text_lower:
            base_action[1] = -0.3
            reasoning = "Turning right"
        elif 'stop' in text_lower:
            base_action = np.zeros(2)
            reasoning = "Stopping"
        
        # Check for completion
        done = any(phrase in text_lower for phrase in [
            'task complete',
            'task finished',
            'goal achieved',
            'mission accomplished',
        ])
        
        # Get current arm positions
        current_state = observation.get('robot_state', {})
        left_arm = current_state.get('left_arm_qpos')
        right_arm = current_state.get('right_arm_qpos')
        
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
        """Generate reasoning using LLaVA."""
        if self.model is None:
            return "Model not loaded"
        
        # Use predict_action for reasoning
        action = self.predict_action(observation, task_description, history)
        return action.reasoning
    
    def _dummy_predict(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> VLMAction:
        """Dummy prediction when model not loaded."""
        # Import from OpenVLA to reuse logic
        from aloha.vlm_openvla import OpenVLAController
        dummy_controller = OpenVLAController.__new__(OpenVLAController)
        return dummy_controller._dummy_predict(observation, task_description, history)
