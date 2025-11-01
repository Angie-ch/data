"""
Helper functions for generating and loading description data for typhoon prediction.
"""

import json
import csv
import os
from typing import Dict, List, Optional, Union
from datetime import datetime


def generate_description_from_metadata(metadata: Dict) -> str:
    """
    Generate CLIP-compatible text descriptions from numerical metadata.
    
    Args:
        metadata: Dictionary containing typhoon metadata
        
    Returns:
        Text description string
    """
    descriptions = []
    
    # Location description
    if 'latitude' in metadata and 'longitude' in metadata:
        desc = f"typhoon center latitude {metadata['latitude']:.2f} longitude {metadata['longitude']:.2f}"
        descriptions.append(desc)
    
    # Intensity description
    if 'wind_speed' in metadata:
        desc = f"maximum sustained wind speed {metadata['wind_speed']} m/s"
        descriptions.append(desc)
    
    if 'pressure' in metadata:
        desc = f"central pressure {metadata['pressure']} hPa"
        descriptions.append(desc)
    
    # Category description
    if 'category' in metadata:
        categories = {
            1: "tropical depression",
            2: "tropical storm",
            3: "severe tropical storm",
            4: "typhoon",
            5: "super typhoon"
        }
        category_text = categories.get(metadata['category'], "tropical cyclone")
        descriptions.append(category_text)
    
    # Movement description
    if 'movement_direction' in metadata and 'movement_speed' in metadata:
        desc = f"moving {metadata['movement_direction']} at {metadata['movement_speed']} km/h"
        descriptions.append(desc)
    
    # Time information
    if 'timestamp' in metadata:
        desc = f"observation at {metadata['timestamp']}"
        descriptions.append(desc)
    
    return " ".join(descriptions)


def load_descriptions_from_json(file_path: str) -> Dict:
    """
    Load description data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary of descriptions
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def load_descriptions_from_csv(file_path: str) -> List[Dict]:
    """
    Load description data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of dictionaries containing descriptions
    """
    descriptions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            descriptions.append(row)
    return descriptions


def save_descriptions_to_json(data: Dict, file_path: str):
    """
    Save description data to JSON file.
    
    Args:
        data: Dictionary of descriptions
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def create_sample_descriptions(output_path: str = "data/typhoon_descriptions.json"):
    """
    Create a sample description file for testing.
    
    Args:
        output_path: Path to save the sample file
    """
    sample_data = {
        "typhoon_001": {
            "timestamp": "2023-08-15T12:00:00Z",
            "description": "category 3 typhoon at latitude 20.5 longitude 130.2 with wind speed 45 m/s",
            "metadata": {
                "latitude": 20.5,
                "longitude": 130.2,
                "wind_speed": 45,
                "pressure": 945,
                "category": 3,
                "movement_direction": "northwest",
                "movement_speed": 15
            }
        },
        "typhoon_002": {
            "timestamp": "2023-08-15T18:00:00Z",
            "description": "severe tropical storm moving northwest at 15 km/h",
            "metadata": {
                "latitude": 21.0,
                "longitude": 129.5,
                "wind_speed": 35,
                "pressure": 970,
                "category": 2,
                "movement_direction": "northwest",
                "movement_speed": 15
            }
        },
        "typhoon_003": {
            "timestamp": "2023-08-16T00:00:00Z",
            "description": "super typhoon with well-defined eye structure",
            "metadata": {
                "latitude": 22.0,
                "longitude": 128.0,
                "wind_speed": 65,
                "pressure": 920,
                "category": 5,
                "movement_direction": "north",
                "movement_speed": 20
            }
        }
    }
    
    save_descriptions_to_json(sample_data, output_path)
    print(f"Sample description file created at: {output_path}")
    print(f"Contains {len(sample_data)} sample typhoon descriptions")


def get_description_for_image(image_id: str, descriptions: Dict) -> Optional[str]:
    """
    Get description text for a specific image ID.
    
    Args:
        image_id: Identifier for the image
        descriptions: Dictionary of descriptions
        
    Returns:
        Description string or None if not found
    """
    if image_id in descriptions:
        if 'description' in descriptions[image_id]:
            return descriptions[image_id]['description']
        elif 'metadata' in descriptions[image_id]:
            return generate_description_from_metadata(descriptions[image_id]['metadata'])
    return None


def batch_generate_descriptions(metadata_list: List[Dict]) -> List[str]:
    """
    Generate descriptions for a batch of metadata entries.
    
    Args:
        metadata_list: List of metadata dictionaries
        
    Returns:
        List of description strings
    """
    return [generate_description_from_metadata(meta) for meta in metadata_list]


if __name__ == "__main__":
    # Create sample description file
    create_sample_descriptions()
    
    # Example: Generate description from metadata
    sample_metadata = {
        "latitude": 20.5,
        "longitude": 130.2,
        "wind_speed": 45,
        "pressure": 945,
        "category": 3,
        "movement_direction": "northwest",
        "movement_speed": 15,
        "timestamp": "2023-08-15T12:00:00Z"
    }
    
    description = generate_description_from_metadata(sample_metadata)
    print(f"\nGenerated description: {description}")
    
    # Example: Load and use descriptions
    try:
        descriptions = load_descriptions_from_json("data/typhoon_descriptions.json")
        print(f"\nLoaded {len(descriptions)} descriptions")
        
        for typhoon_id, data in descriptions.items():
            print(f"\n{typhoon_id}: {data.get('description', 'No description')}")
    except FileNotFoundError:
        print("\nRun this script first to create the sample file, then run again to test loading.")

