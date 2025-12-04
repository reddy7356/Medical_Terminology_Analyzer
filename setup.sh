#!/bin/bash
# Medical Terminology Analyzer Setup Script

echo "🏥 Medical Terminology Analyzer Setup"
echo "====================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
echo "⚙️  Setting up environment..."
cp env.example .env
echo "✅ Environment file created. Please edit .env with your Neon connection string and OpenAI API key."

# Create data directory
mkdir -p data

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials:"
echo "   - NEON_CONNECTION_STRING=your_neon_connection_string"
echo "   - OPENAI_API_KEY=your_openai_api_key"
echo ""
echo "2. Run the system:"
echo "   source venv/bin/activate"
echo "   python medical_terminology_analyzer.py"
echo ""
echo "3. Access web interface: http://localhost:5556"
echo "4. Run MCP server: python terminology_mcp_server.py"





