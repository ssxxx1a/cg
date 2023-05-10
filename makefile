.PHONY : debug_uncond
debug_uncond:
	python -m debugpy --listen 8002 --wait-for-client scripts/classifier_sample.py --classifier_scale 10.0 --classifier_path 256x256_classifier.pt --model_path 256x256_diffusion_uncond.pt --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 4 --num_samples 100 --timestep_respacing 250

.PHONY : debug_cond
debug_cond:
	python -m debugpy --listen 8002 --wait-for-client scripts/classifier_sample.py --classifier_scale 1.0 --classifier_path 256x256_classifier.pt --model_path 256x256_diffusion.pt --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 4 --num_samples 100 --timestep_respacing 250


.PHONY : uncond
uncond:
	python scripts/classifier_sample.py --classifier_scale 10.0 --classifier_path 256x256_classifier.pt --model_path 256x256_diffusion_uncond.pt --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 4 --num_samples 100 --timestep_respacing 250

.PHONY : cond
cond:
	python scripts/classifier_sample.py --classifier_scale 1.0 --classifier_path 256x256_classifier.pt --model_path 256x256_diffusion.pt --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 4 --num_samples 100 --timestep_respacing 250

.PHONY : calc_diff
calc_diff:
	python  calc_diff.py --classifier_scale 10.0 --classifier_path 256x256_classifier.pt --cond_model_path 256x256_diffusion.pt --uncond_model_path 256x256_diffusion_uncond.pt --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 4 --num_samples 100 --timestep_respacing 50

.PHONY : debug_calc_diff
debug_calc_diff:
	python -m debugpy --listen 8002 --wait-for-client calc_diff.py --classifier_scale 10.0 --classifier_path 256x256_classifier.pt --cond_model_path 256x256_diffusion.pt --uncond_model_path 256x256_diffusion_uncond.pt --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --batch_size 4 --num_samples 100 --timestep_respacing 250


.PHONY : clean
clean:
	rm -rf openai/*