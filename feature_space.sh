cd src

energy_function=energy_function_feature
inference_function=CGMTL_inference_feature
gnn_energy_model=GNN_Energy_Model_1st

seed_list=(0 1 2 3 4)

for seed in "${seed_list[@]}"; do

        mtl_method=gnn
        dataset=chembl_dense_10
        output_folder=../feature_checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_model_file="$output_folder"/model

        date

        python main_feature.py \
        --with_edge_attention=True \
        --lmd_1=0.1 \
        --use_trivial=True \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" \
        --inference_function="$inference_function" \
        --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=50 \
        --ebm_GNN_layer_num=3 \
        --output_model_file="$output_model_file" \
        --epochs=300 \
        --batch_size=16 \
        --PPI_pretrained_epochs=100 \
        --MTL_pretrained_epochs=100 \
        --seed="$seed"


done

for seed in "${seed_list[@]}"; do

        mtl_method=gnn
        dataset=chembl_dense_50
        output_folder=../feature_checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_model_file="$output_folder"/model

        date

        python main_feature.py \
        --with_edge_attention=True \
        --lmd_1=0.5 \
        --use_trivial=True \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" \
        --inference_function="$inference_function" \
        --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --output_model_file="$output_model_file" \
        --epochs=300 \
        --batch_size=64 \
        --PPI_pretrained_epochs=100 \
        --MTL_pretrained_epochs=100 \
        --seed="$seed"


done


for seed in "${seed_list[@]}"; do

        mtl_method=gnn
        dataset=chembl_dense_100
        output_folder=../feature_checkpoint/"$mtl_method"/"$dataset"/"$seed"
        mkdir -p "$output_folder"
        output_model_file="$output_folder"/model

        date

        python main_feature.py \
        --with_edge_attention=True \
        --lmd_1=0.5 \
        --use_trivial=True \
        --mtl_method=structured_prediction \
        --dataset="$dataset" \
        --energy_function="$energy_function" \
        --inference_function="$inference_function" \
        --gnn_energy_model="$gnn_energy_model" \
        --task_emb_dim=50 \
        --PPI_threshold=0.1 \
        --ebm_GNN_dim=100 \
        --ebm_GNN_layer_num=3 \
        --output_model_file="$output_model_file" \
        --epochs=300 \
        --batch_size=64 \
        --PPI_pretrained_epochs=100 \
        --MTL_pretrained_epochs=100 \
        --seed="$seed"


done















