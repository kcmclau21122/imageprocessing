"""
LLM integration module for enhanced face identification
Supports Ollama (LLaVA) and OpenAI GPT-4 Vision for additional context
"""
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from io import BytesIO
from PIL import Image

import config

logger = logging.getLogger(__name__)


class LLMVerifier:
    """
    LLM-based verifier for face identification using vision models.
    Can provide additional context and verification for ambiguous cases.
    """
    
    def __init__(self, provider: str = 'ollama'):
        """
        Initialize LLM verifier.
        
        Args:
            provider: LLM provider ('ollama' or 'openai')
        """
        self.provider = provider.lower()
        
        if self.provider == 'ollama':
            try:
                import ollama
                self.client = ollama
                self.model = config.OLLAMA_MODEL
                logger.info(f"Initialized Ollama with model: {self.model}")
            except ImportError:
                logger.error("Ollama not installed. Install with: pip install ollama")
                self.client = None
        
        elif self.provider == 'openai':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.model = "gpt-4o"  # GPT-4 with vision
                logger.info("Initialized OpenAI client")
            except ImportError:
                logger.error("OpenAI not installed. Install with: pip install openai")
                self.client = None
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for API calls.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def verify_with_ollama(self, image_path: str, detected_names: List[str]) -> Dict:
        """
        Verify detected faces using Ollama LLaVA model.
        
        Args:
            image_path: Path to the image
            detected_names: List of names detected by the face recognizer
            
        Returns:
            Dictionary with verification results
        """
        if not self.client:
            return {'success': False, 'error': 'Ollama client not initialized'}
        
        try:
            # Create prompt
            if detected_names:
                names_str = ", ".join(detected_names)
                prompt = (
                    f"This image has been analyzed by a face recognition system. "
                    f"The system detected the following people: {names_str}. "
                    f"Please describe what you see in the image, particularly focusing on people. "
                    f"How many people do you see? Do the detected names seem reasonable?"
                )
            else:
                prompt = (
                    "Describe the people you see in this image. "
                    "How many people are there? Can you see their faces clearly?"
                )
            
            # Call Ollama
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            
            verification_text = response['message']['content']
            
            return {
                'success': True,
                'verification': verification_text,
                'provider': 'ollama',
                'model': self.model
            }
            
        except Exception as e:
            logger.error(f"Ollama verification error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def verify_with_openai(self, image_path: str, detected_names: List[str]) -> Dict:
        """
        Verify detected faces using OpenAI GPT-4 Vision.
        
        Args:
            image_path: Path to the image
            detected_names: List of names detected by the face recognizer
            
        Returns:
            Dictionary with verification results
        """
        if not self.client:
            return {'success': False, 'error': 'OpenAI client not initialized'}
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Create prompt
            if detected_names:
                names_str = ", ".join(detected_names)
                prompt = (
                    f"This image has been analyzed by a face recognition system. "
                    f"The system detected the following people: {names_str}. "
                    f"Please describe what you see in the image, particularly focusing on people. "
                    f"How many people do you see? Do the detected names seem reasonable?"
                )
            else:
                prompt = (
                    "Describe the people you see in this image. "
                    "How many people are there? Can you see their faces clearly?"
                )
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            verification_text = response.choices[0].message.content
            
            return {
                'success': True,
                'verification': verification_text,
                'provider': 'openai',
                'model': self.model
            }
            
        except Exception as e:
            logger.error(f"OpenAI verification error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def verify_identification(self, image_path: str, 
                            detected_names: List[str]) -> Dict:
        """
        Verify face identification using configured LLM provider.
        
        Args:
            image_path: Path to the image
            detected_names: List of names detected by the face recognizer
            
        Returns:
            Dictionary with verification results
        """
        if self.provider == 'ollama':
            return self.verify_with_ollama(image_path, detected_names)
        elif self.provider == 'openai':
            return self.verify_with_openai(image_path, detected_names)
        else:
            return {'success': False, 'error': 'Invalid provider'}
    
    def get_image_description(self, image_path: str) -> Optional[str]:
        """
        Get a general description of the image from the LLM.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Description text or None if failed
        """
        result = self.verify_identification(image_path, [])
        
        if result.get('success'):
            return result.get('verification')
        
        return None


class EnhancedIdentifier:
    """
    Enhanced face identifier that combines traditional face recognition
    with LLM verification for improved accuracy.
    """
    
    def __init__(self, face_identifier, llm_provider: str = 'ollama'):
        """
        Initialize enhanced identifier.
        
        Args:
            face_identifier: Traditional FaceIdentifier instance
            llm_provider: LLM provider for verification
        """
        self.face_identifier = face_identifier
        self.llm_verifier = LLMVerifier(llm_provider)
    
    def identify_with_verification(self, image_path: str, 
                                   confidence_threshold: float = 0.75) -> Dict:
        """
        Identify faces with LLM verification.
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for identification
            
        Returns:
            Dictionary with identification and verification results
        """
        # First, use traditional face recognition
        face_results = self.face_identifier.identify_faces(
            image_path, confidence_threshold
        )
        
        # Extract detected names
        detected_names = [
            r['predicted_label'] for r in face_results 
            if r['identified']
        ]
        
        # Get LLM verification
        verification = self.llm_verifier.verify_identification(
            image_path, detected_names
        )
        
        return {
            'face_recognition': face_results,
            'llm_verification': verification,
            'detected_names': detected_names
        }
    
    def batch_identify_with_verification(self, image_paths: List[str],
                                        confidence_threshold: float = 0.75) -> Dict:
        """
        Identify faces in multiple images with verification.
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence for identification
            
        Returns:
            Dictionary mapping image paths to results
        """
        from tqdm import tqdm
        
        results = {}
        
        for image_path in tqdm(image_paths, desc="Identifying with verification"):
            result = self.identify_with_verification(image_path, confidence_threshold)
            results[image_path] = result
        
        return results


def test_llm_connection(provider: str = 'ollama'):
    """
    Test LLM connection and capabilities.
    
    Args:
        provider: LLM provider to test
    """
    print(f"\nTesting {provider.upper()} connection...")
    
    try:
        verifier = LLMVerifier(provider)
        
        if verifier.client is None:
            print(f"✗ {provider.upper()} client failed to initialize")
            return False
        
        print(f"✓ {provider.upper()} client initialized successfully")
        print(f"  Model: {verifier.model}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {provider}: {str(e)}")
        return False


if __name__ == "__main__":
    """Test LLM connections."""
    print("="*60)
    print("LLM INTEGRATION TEST")
    print("="*60)
    
    # Test Ollama
    test_llm_connection('ollama')
    
    # Test OpenAI if API key is set
    if config.OPENAI_API_KEY:
        print()
        test_llm_connection('openai')
    else:
        print("\nOpenAI API key not set, skipping OpenAI test")
    
    print("="*60)