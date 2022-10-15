# reflection_removal_inference

Please refer to the original repository: https://github.com/mingcv/YTMT-Strategy

I have updated the import statements as per the new skimage version

I have also added the test_image.py which performs inference over several images kept in a folder.

## Execution steps
```
python test_image.py --gpu_ids -1 --dataset_path <path where all the images are kept> --output_path <output path>
```

Please download the trained weights from: https://drive.google.com/file/d/1yOKFzhhFUdbKzU3eafYKFLN7AdHqW4_7/view?pli=1
And keep it inside the current folder as: ./checkpoints/ytmt_uct_sirs_test/ytmt_uct_sirs_68_077_00595364.pt
