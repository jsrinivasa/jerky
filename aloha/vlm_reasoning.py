#!/usr/bin/env python3

"""
Closed-Loop VLM Reasoning Controller for Mobile ALOHA

This module implements a closed-loop reasoning system that uses
vision-language models (like GPT-4V, Claude, or similar) to:
1. Observe the environment through cameras
2. Reason about the current state and task
3. Plan and execute actions
4. Monitor progress and adapt

This provides a more flexible alternative to OpenVLA using API-based VLMs.
"""

import json
import base64
import io
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image

from aloha.vlm_controller import VLMController, VLMAction
from aloha.constants import START_ARM_POSE


class ReasoningVLMController(VLMController):
    """
    Closed-loop reasoning controller using vision-language models.
    
    This controller maintains a reasoning loop:
    1. Observe: Get visual and state information
    2. Reason: Understand current situation relative to goal
    3. Plan: Decide what action to take
    4. Act: Execute the action
    5. Reflect: Evaluate progress
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4-vision-preview",
        api_provider: str = "openai",
        verbose_logging: bool = True,
        **kwargs
    ):
        """
        Initialize reasoning VLM controller.
        
        Args:
            api_key: API key for the VLM service
            model_name: Name of the model to use
            api_provider: Provider ('openai', 'anthropic', 'local')
            verbose_logging: Enable detailed input/output logging
            **kwargs: Additional arguments for VLMController
        """
        super().__init__(**kwargs)
        
        self.api_key = api_key
        self.model_name = model_name
        self.api_provider = api_provider
        self.verbose_logging = verbose_logging
        self.step_counter = 0
        
        # Initialize API client
        self.client = None
        if self.api_provider == "openai" and api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                print(f"Initialized OpenAI client with model: {model_name}")
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
        elif self.api_provider == "anthropic" and api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                print(f"Initialized Anthropic client with model: {model_name}")
            except ImportError:
                print("Warning: anthropic package not installed. Install with: pip install anthropic")
        else:
            print("Warning: Running in rule-based mode without VLM API")
        
        # Reasoning state
        self.task_plan = []
        self.current_subtask = None
        self.reflection_history = []
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> VLMAction:
        """
        Predict action using closed-loop reasoning.
        
        Args:
            observation: Current observation
            task_description: Natural language task
            history: Execution history
            
        Returns:
            VLMAction with reasoning
        """
        self.step_counter += 1
        
        if self.verbose_logging:
            print(f"\n{'='*70}")
            print(f"[VLM REASONING - Step {self.step_counter}]")
            print(f"{'='*70}")
            print(f"Task: \"{task_description}\"")
            print(f"History length: {len(history)} steps")
        
        # Generate reasoning about current state
        reasoning = self.reason_about_task(observation, task_description, history)
        
        if self.verbose_logging:
            print(f"\n[REASONING OUTPUT]")
            print(f"{reasoning[:500]}{'...' if len(reasoning) > 500 else ''}")
        
        # Get action from reasoning
        action = self._reasoning_to_action(
            reasoning,
            observation,
            task_description,
            history
        )
        
        if self.verbose_logging:
            print(f"\n[PARSED ACTION]")
            print(f"Base action: linear={action.base_action[0]:.3f}, angular={action.base_action[1]:.3f}")
            print(f"Confidence: {action.confidence:.2f}")
            print(f"Done: {action.done}")
            print(f"{'='*70}\n")
        
        return action
    
    def reason_about_task(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """
        Generate detailed reasoning about the current state.
        
        Args:
            observation: Current observation
            task_description: Task description
            history: Execution history
            
        Returns:
            Reasoning text
        """
        if self.client is None:
            # Fallback to rule-based reasoning
            return self._rule_based_reasoning(observation, task_description, history)
        
        try:
            # Prepare image for API
            images = observation['images']
            image = images.get('cam_high') or images.get('cam_left_wrist')
            
            if image is None:
                return "No visual input available"
            
            # Encode image
            image_data = self._encode_image(image)
            
            # Create reasoning prompt
            prompt = self._create_reasoning_prompt(
                task_description,
                observation,
                history
            )
            
            # Call VLM API
            reasoning = self._call_vlm_api(prompt, image_data)
            
            return reasoning
            
        except Exception as e:
            print(f"Error in reasoning: {e}")
            return f"Error: {str(e)}"
    
    def _create_reasoning_prompt(
        self,
        task_description: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:
        """Create a detailed prompt for the VLM."""
        step_num = len(history)
        
        # Get robot state info
        state_info = ""
        if 'robot_state' in observation:
            state = observation['robot_state']
            state_info = f"""
Current Robot State:
- Left arm position: {np.array2string(state.get('left_arm_qpos', []), precision=2)}
- Right arm position: {np.array2string(state.get('right_arm_qpos', []), precision=2)}
- Left gripper: {state.get('left_gripper_qpos', 0):.3f}
- Right gripper: {state.get('right_gripper_qpos', 0):.3f}
"""
        
        # Recent history summary
        history_summary = ""
        if len(history) > 0:
            recent_actions = history[-3:]  # Last 3 actions
            history_summary = "\nRecent Actions:\n"
            for h in recent_actions:
                action = h.get('action')
                if action:
                    history_summary += f"- Step {h['step']}: {action.reasoning}\n"
        
        prompt = f"""You are controlling a Mobile ALOHA robot. This robot has:
- A mobile base that can move and rotate
- Two 6-DOF robotic arms (left and right)
- Two grippers for manipulation

Task: {task_description}

Current Step: {step_num + 1}
{state_info}
{history_summary}

Based on the image and current state, provide:
1. What you observe in the scene
2. Current progress toward the task goal
3. What action should be taken next
4. Whether the task is complete

Respond in JSON format:
{{
    "observation": "What you see in the image",
    "progress": "Assessment of progress (0-100%)",
    "next_action": {{
        "type": "base_move|arm_move|grasp|done",
        "description": "What to do",
        "base": {{"linear": 0.0, "angular": 0.0}},
        "confidence": 0.8
    }},
    "reasoning": "Your reasoning process",
    "task_complete": false
}}
"""
        return prompt
    
    def _call_vlm_api(self, prompt: str, image_data: str) -> str:
        """Call the VLM API with prompt and image."""
        if self.api_provider == "openai":
            return self._call_openai(prompt, image_data)
        elif self.api_provider == "anthropic":
            return self._call_anthropic(prompt, image_data)
        else:
            return "API provider not configured"
    
    def _call_openai(self, prompt: str, image_data: str) -> str:
        """Call OpenAI API (GPT-4V)."""
        try:
            if self.verbose_logging:
                print(f"\n[CALLING OPENAI API]")
                print(f"Model: {self.model_name}")
                print(f"Prompt length: {len(prompt)} chars")
                print(f"Image data length: {len(image_data)} chars (base64)")
                print(f"Sending request...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
            )
            
            result = response.choices[0].message.content
            
            if self.verbose_logging:
                print(f"[OPENAI RESPONSE]")
                print(f"Response length: {len(result)} chars")
                print(f"Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
            
            return result
        except Exception as e:
            if self.verbose_logging:
                print(f"[OPENAI ERROR]: {str(e)}")
            return f"OpenAI API error: {str(e)}"
    
    def _call_anthropic(self, prompt: str, image_data: str) -> str:
        """Call Anthropic API (Claude)."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API error: {str(e)}"
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def _reasoning_to_action(
        self,
        reasoning: str,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> VLMAction:
        """
        Convert reasoning text to executable action.
        
        Parses the VLM's reasoning output and generates robot commands.
        """
        try:
            # Try to parse JSON response
            if "```json" in reasoning:
                json_str = reasoning.split("```json")[1].split("```")[0]
            elif "{" in reasoning and "}" in reasoning:
                start = reasoning.find("{")
                end = reasoning.rfind("}") + 1
                json_str = reasoning[start:end]
            else:
                # Fallback: use text-based parsing
                return self._parse_text_reasoning(reasoning, observation)
            
            response = json.loads(json_str)
            
            # Extract action components
            next_action = response.get('next_action', {})
            base_cmd = next_action.get('base', {'linear': 0.0, 'angular': 0.0})
            confidence = next_action.get('confidence', 0.5)
            task_complete = response.get('task_complete', False)
            reasoning_text = response.get('reasoning', reasoning)
            
            # Build action
            base_action = np.array([
                float(base_cmd.get('linear', 0.0)),
                float(base_cmd.get('angular', 0.0))
            ])
            
            # Get current arm positions (maintain for now)
            current_state = observation.get('robot_state', {})
            left_arm = current_state.get('left_arm_qpos')
            right_arm = current_state.get('right_arm_qpos')
            
            return VLMAction(
                base_action=base_action,
                left_arm_action=left_arm,
                right_arm_action=right_arm,
                reasoning=reasoning_text,
                confidence=confidence,
                done=task_complete,
            )
            
        except Exception as e:
            print(f"Error parsing reasoning: {e}")
            return self._parse_text_reasoning(reasoning, observation)
    
    def _parse_text_reasoning(
        self,
        reasoning: str,
        observation: Dict[str, Any],
    ) -> VLMAction:
        """Fallback text-based parsing of reasoning."""
        text_lower = reasoning.lower()
        
        base_action = np.array([0.0, 0.0])
        
        # Simple keyword-based parsing
        if 'forward' in text_lower or 'move forward' in text_lower:
            base_action[0] = 0.2
        elif 'backward' in text_lower or 'move backward' in text_lower:
            base_action[0] = -0.2
        elif 'turn left' in text_lower or 'rotate left' in text_lower:
            base_action[1] = 0.3
        elif 'turn right' in text_lower or 'rotate right' in text_lower:
            base_action[1] = -0.3
        elif 'stop' in text_lower:
            base_action = np.array([0.0, 0.0])
        
        done = any(word in text_lower for word in 
                   ['complete', 'done', 'finished', 'success'])
        
        # Get current arm positions
        current_state = observation.get('robot_state', {})
        left_arm = current_state.get('left_arm_qpos')
        right_arm = current_state.get('right_arm_qpos')
        
        return VLMAction(
            base_action=base_action,
            left_arm_action=left_arm,
            right_arm_action=right_arm,
            reasoning=reasoning[:200],  # Truncate for display
            confidence=0.5,
            done=done,
        )
    
    def _rule_based_reasoning(
        self,
        observation: Dict[str, Any],
        task_description: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """
        Fallback rule-based reasoning when no API is available.
        """
        step = len(history)
        
        reasoning = f"Step {step + 1}: "
        
        # Simple state machine for demo purposes
        if 'move forward' in task_description.lower():
            if step < 10:
                reasoning += "Moving forward as requested"
            else:
                reasoning += "Forward movement complete"
        elif 'turn' in task_description.lower():
            if step < 8:
                reasoning += "Executing turn maneuver"
            else:
                reasoning += "Turn complete"
        elif 'explore' in task_description.lower():
            cycle = step % 20
            if cycle < 5:
                reasoning += "Moving forward to explore"
            elif cycle < 10:
                reasoning += "Rotating to scan environment"
            elif cycle < 15:
                reasoning += "Moving to new position"
            else:
                reasoning += "Analyzing explored area"
        else:
            # Generic exploration
            if step < 5:
                reasoning += "Initializing task execution"
            elif step < 15:
                reasoning += "Executing primary task actions"
            else:
                reasoning += "Task appears complete"
        
        return reasoning
