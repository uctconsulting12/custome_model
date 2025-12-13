from ultralytics.models.sam import SAM2VideoPredictor

# Create video predictor
predictor = SAM2VideoPredictor(model="sam2-t.pt", imgsz=1024, conf=0.25)

# Track all instances of a concept
results = predictor(source="ppe_detection.mp4", prompt="person")

for r in results:
    print(r)

# # Run inference with multiple points
# results = predictor(source="test.mp4", points=[[920, 470], [909, 138]], labels=[1, 1])

# # Run inference with multiple points prompt per object
# results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

# # Run inference with negative points prompt
# results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])