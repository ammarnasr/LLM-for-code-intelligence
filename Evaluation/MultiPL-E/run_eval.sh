# 
echo "Running eval for 350M multi-humaneval java pass at 10 gen config"

podman run --rm --network none -v ./tgt/codegen_350M_multi_humaneval_java_pass_at_10_gen_config:/tgt/codegen_350M_multi_humaneval_java_pass_at_10_gen_config:rw multipl-e-eval --dir /tgt/codegen_350M_multi_humaneval_java_pass_at_10_gen_config --output-dir /tgt/codegen_350M_multi_humaneval_java_pass_at_10_gen_config --recursive