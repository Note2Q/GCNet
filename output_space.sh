cd src

energy_function=energy_function_output
inference_function=CGMTL_inference_output
gnn_energy_model=GNN_Energy_Model_2nd

seed_list=(0 1 2 3 4)


for seed in "${seed_list[@]}"; do

        mtl_method=causal_ebm
        dataset=chembl_dense_10
        output_folder=../output_checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_model_file="$output_folder"/model
        date

        python main_output.py \
        --with_edge_attention=True\
        --NCE_mode=do_operation \
        --epochs=300 \
        --lmd_2=0.05 \
        --lmd_3=0.05 \
        --use_trivial=False \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" \
        --inference_function="$inference_function" \
        --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --filling_mode=gnn \
        --GS_iteration=2 \
        --structured_lambda=0.1 \
        --output_model_file="$output_model_file" \
        --seed="$seed"\
        --batch_size=16 \
        --PPI_pretrained_epochs=100 \
        --MTL_pretrained_epochs=100 


done

for seed in "${seed_list[@]}"; do

        mtl_method=causal_ebm
        dataset=chembl_dense_50
        output_folder=../output_checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_model_file="$output_folder"/model
        date

        python main_output.py \
        --with_edge_attention=True\
        --NCE_mode=do_operation \
        --epochs=300 \
        --lmd_2=0.10 \
        --lmd_3=0.10 \
        --use_trivial=False \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" \
        --inference_function="$inference_function" \
        --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --filling_mode=gnn \
        --GS_iteration=2 \
        --structured_lambda=0.1 \
        --output_model_file="$output_model_file" \
        --seed="$seed"\
        --batch_size=64 \
        --PPI_pretrained_epochs=100 \
        --MTL_pretrained_epochs=100 

done

for seed in "${seed_list[@]}"; do

        mtl_method=causal_ebm
        dataset=chembl_dense_100
        output_folder=../output_checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_model_file="$output_folder"/model
        date

        python main_output.py \
        --with_edge_attention=True\
        --NCE_mode=do_operation \
        --epochs=300 \
        --lmd_2=0.10 \
        --lmd_3=0.10 \
        --use_trivial=False \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" \
        --inference_function="$inference_function" \
        --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --filling_mode=gnn \
        --GS_iteration=2 \
        --structured_lambda=0.1 \
        --output_model_file="$output_model_file" \
        --seed="$seed"\
        --batch_size=64 \
        --PPI_pretrained_epochs=100 \
        --MTL_pretrained_epochs=100 

done
