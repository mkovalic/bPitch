model = BasicPitchModel()
sample_input = torch.randn(4, 8, 224, 908)  # Batch size 4
yo, yp, yn = model(sample_input)
print(f"Yo Shape: {yo.shape}, Yp Shape: {yp.shape}, Yn Shape: {yn.shape}")

