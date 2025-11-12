import os
import argparse
import json
import torch
from omegaconf import OmegaConf

from dataset.dentition_data import DentitionDataset, FDIS_ORDER
from model.bound_encoder import BoundEncoder
from dataset.cylinder_util import unnormalize_bound, create_cylinder_trimesh, export_dentition

def main(cfg):

    os.makedirs(cfg.inference.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DentitionDataset(
        path=cfg.inference.datapath,
        mode='test',
        tooth_npoints=cfg.data.tooth_npoints
    )

    checkpoint_path = cfg.inference.model_ckpt
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    model = BoundEncoder.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    patient_id_list = test_dataset.get_patient_ids()

    # patient_id_list: for test dataset 
    # {
    #     "TADP0046":[[11,12],[45,46,47]], # list of missing teeth FDIs to generate for patient TADP0046
    #     "TADP0047":[[33],[15,13,11,45,26,27]]
    # }


    for patient_id in patient_id_list:


        patient_save_dir = os.path.join(cfg.inference.save_dir, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)

        sample = test_dataset.sample_patient_by_patient_id(patient_id)
        dentition_points = sample['dentition_points'].unsqueeze(0).to(device)
        
        shift = sample['shift'].numpy()
        scale = sample['scale'].numpy()

        dentition_points_np = dentition_points.detach().cpu().numpy()
        missing_fdis_to_generate = patient_id_list[patient_id]

        if missing_fdis_to_generate != None:
            for missing_fdis in missing_fdis_to_generate:      

                print(f"Generating for bounds for patient {patient_id} with missing FDIs: {missing_fdis}")

                scenario_save_dir = os.path.join(patient_save_dir, f'scenario_{missing_fdis}')
                os.makedirs(scenario_save_dir, exist_ok=True)

                obs_mask = torch.ones_like(dentition_points[:, :, :1, :1]).to(device)
                missing_indices = [FDIS_ORDER.index(fdi) for fdi in missing_fdis]
                obs_mask[0, missing_indices, 0, 0] = 0


                with torch.no_grad():
                    pred_bound = model(dentition_points, obs_mask)[0].detach().cpu().numpy()

                    for idx,fdi in enumerate(FDIS_ORDER):
                        if obs_mask[0][idx] == 0:
                            cyl_param = unnormalize_bound(pred_bound[idx], shift, scale).squeeze()

                            bounding_dict = {
                                'cylinder':{'cx':cyl_param[0],'cy':cyl_param[1],'cz':cyl_param[2],'h':cyl_param[3],'r':cyl_param[4]},
                            }
                            
                            tooth_name = f'{patient_id}_[{fdi}]_gen_predbound.json'
                            
                            with open(os.path.join(scenario_save_dir, tooth_name), "w") as file:
                                json.dump(bounding_dict, file)

                            mesh_obj = create_cylinder_trimesh(cyl_param[0],cyl_param[1],cyl_param[2],cyl_param[3],cyl_param[4])
                            mesh_obj.export(os.path.join(scenario_save_dir, tooth_name.replace('.json','.obj')))
                    
                    obs_mask_np = obs_mask.detach().cpu()
                    condition_points = dentition_points_np[0][obs_mask_np.view(-1).bool()]
                    export_dentition(patient_id, scenario_save_dir, condition_points, shift, scale, combine_teeth=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/boundpred_cfg.yaml'
    )
    cli_args = parser.parse_args()

    main(OmegaConf.load(cli_args.config))