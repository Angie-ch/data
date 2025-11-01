# Description Data Sources for Typhoon Prediction

This document outlines where to find text description/metadata data for typhoon prediction tasks, especially for use with CLIP alignment.

## Official Data Sources

### 1. **IBTrACS (International Best Track Archive for Climate Stewardship)**
- **URL**: https://www.ncei.noaa.gov/products/international-best-track-archive
- **Description**: Global tropical cyclone best track data with metadata
- **Data Format**: CSV, NetCDF, JSON
- **Contains**: Storm names, tracks, intensity, pressure, wind speed, location

### 2. **Joint Typhoon Warning Center (JTWC)**
- **URL**: https://www.metoc.navy.mil/jtwc/jtwc.html
- **Description**: Detailed typhoon reports and bulletins
- **Data Format**: Text reports, bulletins
- **Contains**: Written descriptions, intensity classifications, forecasts

### 3. **Japan Meteorological Agency (JMA)**
- **URL**: https://www.jma.go.jp/jma/indexe.html
- **Description**: Typhoon advisories and bulletins
- **Data Format**: Text bulletins, XML
- **Contains**: Detailed weather descriptions, classifications

### 4. **RAMMB (Regional and Mesoscale Meteorology Branch)**
- **URL**: https://www.rammb.cira.colostate.edu/
- **Description**: Satellite imagery with metadata
- **Data Format**: Images with associated metadata files
- **Contains**: Image descriptions, storm classifications

### 5. **Himawari-8 Satellite Data**
- **Description**: Satellite imagery with timestamps and location data
- **Metadata**: Can be extracted from image filenames and headers
- **Contains**: Time, location, channel information

## Description Data Format for CLIP Integration

### Example Description Formats

```python
# Format 1: Structured metadata as text
descriptions = [
    "typhoon center latitude 20.5 longitude 130.2 wind speed 45 m/s pressure 945 hPa",
    "tropical cyclone intensity category 3 moving northwest at 15 km/h",
    "severe tropical storm with maximum sustained winds of 55 m/s"
]

# Format 2: Natural language descriptions
descriptions = [
    "A powerful typhoon with a well-defined eye structure",
    "Tropical cyclone showing rapid intensification",
    "Typhoon approaching landfall with strong winds and heavy rainfall"
]

# Format 3: Time-series descriptions
descriptions = [
    f"observation at {timestamp}: typhoon at lat {lat:.2f} lon {lon:.2f}, wind {wind} m/s",
    f"time step {t}: category {category} cyclone moving {direction}"
]
```

## Data Structure

### Recommended JSON Format
```json
{
  "typhoon_id": "TY2023-15",
  "timestamp": "2023-08-15T12:00:00Z",
  "descriptions": [
    {
      "type": "location",
      "text": "typhoon center latitude 20.5 longitude 130.2"
    },
    {
      "type": "intensity",
      "text": "category 3 typhoon with maximum wind speed 45 m/s"
    },
    {
      "type": "movement",
      "text": "moving northwest at 15 km/h"
    },
    {
      "type": "natural",
      "text": "well-defined eye structure with strong convection"
    }
  ],
  "metadata": {
    "pressure": 945,
    "wind_speed": 45,
    "category": 3,
    "latitude": 20.5,
    "longitude": 130.2
  }
}
```

## Generating Descriptions from Metadata

### From Numerical Data
You can generate text descriptions from numerical weather data:

```python
def generate_description_from_metadata(metadata):
    """Generate CLIP-compatible text descriptions from numerical metadata."""
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
        categories = {1: "tropical depression", 2: "tropical storm", 
                     3: "severe tropical storm", 4: "typhoon", 5: "super typhoon"}
        category_text = categories.get(metadata['category'], "tropical cyclone")
        descriptions.append(category_text)
    
    # Movement description
    if 'movement_direction' in metadata and 'movement_speed' in metadata:
        desc = f"moving {metadata['movement_direction']} at {metadata['movement_speed']} km/h"
        descriptions.append(desc)
    
    return " ".join(descriptions)
```

## Integration with Your Data Loaders

### Example Implementation
```python
# In data_loaders/himawari8_loader.py or dataset.py
class TyphoonDataset:
    def __init__(self, image_dir, metadata_file):
        self.image_dir = image_dir
        self.metadata = self.load_metadata(metadata_file)
    
    def load_metadata(self, metadata_file):
        # Load metadata from JSON/CSV/etc.
        # This should contain description data
        pass
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        
        # Get description from metadata
        description = self.generate_description(idx)
        
        return {
            'image': image,
            'description': description,
            'metadata': self.metadata[idx]
        }
    
    def generate_description(self, idx):
        meta = self.metadata[idx]
        return generate_description_from_metadata(meta)
```

## Where to Store Description Data

1. **JSON files**: Store alongside image files
   ```
   data/
   ├── images/
   │   ├── typhoon_2023_08_15_12_00.png
   │   └── ...
   └── descriptions/
       ├── typhoon_2023_08_15_12_00.json
       └── ...
   ```

2. **CSV files**: Single CSV with all descriptions
   ```
   data/
   ├── images/
   └── typhoon_descriptions.csv
   ```

3. **Embedded in image metadata**: EXIF or other metadata headers

4. **Database**: SQLite or PostgreSQL for structured queries

## Quick Start: Creating Sample Description Data

```python
import json
import os

def create_sample_description_file(output_path):
    """Create a sample description file for testing."""
    sample_data = {
        "typhoon_001": {
            "timestamp": "2023-08-15T12:00:00Z",
            "description": "category 3 typhoon at latitude 20.5 longitude 130.2 with wind speed 45 m/s",
            "metadata": {
                "latitude": 20.5,
                "longitude": 130.2,
                "wind_speed": 45,
                "pressure": 945,
                "category": 3
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
                "category": 2
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample description file created at: {output_path}")

# Usage
if __name__ == "__main__":
    create_sample_description_file("data/typhoon_descriptions.json")
```

## Next Steps

1. **Download data** from IBTrACS or JTWC
2. **Extract metadata** from your existing data sources (check `Datadownload/` directory)
3. **Generate descriptions** from numerical metadata using the helper function above
4. **Implement data loaders** in `data_loaders/dataset.py` to load descriptions with images
5. **Test CLIP alignment** using `clip_integration.py` with your real data

## Checking Your Existing Data

You may already have description data in:
- `Datadownload/` - Check for metadata files
- `Two_Tower_Model/data_cache/` - May contain structured data
- `diffusion/data/` - Check for metadata files
- `corrdiff/data/` - May have associated metadata

Run this to check:
```bash
find . -name "*.json" -o -name "*.csv" -o -name "*metadata*" | grep -i typhoon
```

