#!/bin/bash

# Configuration
EC2_USER="ubuntu"  # Change this to your EC2 username
EC2_HOST="ec2-35-175-223-63.compute-1.amazonaws.com"  # Change this to your EC2 instance
KEY_PATH="Yowla.pem"  # Change this to your key path
REMOTE_DIR="/home/ubuntu/MayesMT5"  # Change this to your desired remote directory

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting EC2 training setup...${NC}"

# Create remote directory
echo "Creating remote directory..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_HOST" "mkdir -p $REMOTE_DIR"

# Transfer files
echo "Transferring files to EC2..."
scp -i "$KEY_PATH" -r \
    config.py \
    game_environment.py \
    train_ec2.py \
    requirements.txt \
    "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

# Install dependencies and run training
echo "Setting up environment and starting training..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_HOST" "cd $REMOTE_DIR && \
    python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt && \
    python3 train_ec2.py"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    
    # Create local models directory if it doesn't exist
    mkdir -p models
    
    # Download trained model and logs
    echo "Downloading trained model and logs..."
    scp -i "$KEY_PATH" -r "$EC2_USER@$EC2_HOST:$REMOTE_DIR/models/*" ./models/
    scp -i "$KEY_PATH" -r "$EC2_USER@$EC2_HOST:$REMOTE_DIR/training_*.log" ./
    
    echo -e "${GREEN}Files downloaded successfully!${NC}"
else
    echo -e "${RED}Training failed! Check the logs for details.${NC}"
fi 