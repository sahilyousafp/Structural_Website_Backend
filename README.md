# GLB to 3DM Structural Analysis Flask API

This Flask application processes GLB files from Supabase and performs structural analysis, generating combined JSON output with nodes, columns, and beams data.

## Features

- Loads GLB files directly from Supabase (no local file storage)
- Converts GLB to 3DM in memory
- Performs structural analysis and generates grid-based columns and beams
- Outputs combined JSON to Supabase "analysis-results" bucket
- REST API endpoint for processing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the server:
```bash
python flask_app.py
```

The server will run on `http://localhost:5000`

### API Endpoint:

**POST** `/process`

Processes the latest GLB file from the configured Supabase bucket and returns structural analysis results.

**Response:**
```json
{
    "status": "success",
    "message": "Processing complete", 
    "filename": "filename_structural_data.json",
    "nodes_count": 123,
    "columns_count": 45,
    "beams_count": 67
}
```

### Example usage:
```bash
curl -X POST http://localhost:5000/process
```

## Configuration

The application is configured to:
- Use Supabase bucket "models" for input GLB files
- Use folder "79edaed4-a719-4390-a485-519b68fa68ea" 
- Upload results to "analysis-results" bucket
- Process the most recently uploaded GLB file

## Output

The application generates a combined JSON file containing:
- **metadata**: Processing parameters and timestamps
- **nodes**: 3D coordinate points
- **columns**: Vertical structural elements
- **beams**: Horizontal structural elements

All output is uploaded to the Supabase "analysis-results" bucket.
