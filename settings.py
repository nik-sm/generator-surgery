"""Experiment Settings"""
n_measure_64 = [100, 150, 200, 300, 400, 600, 1000, 2000, 3000, 4000, 6000]
n_measure_128 = [
    400, 600, 800, 1200, 1600, 2400, 4000, 8000, 12000, 16000, 24000
]

forward_models = {
    'iagan_began_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'began_opt_error_fake_imgs': {
        'NoOp': [{}],
    },
    'vanilla_vae_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'vanilla_vae': {
        # 'NoOp': [{}],  # Note - z_lr=1, steps=40 for NoOp
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'beta_vae_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'beta_vae': {
        # 'NoOp': [{}],  # Note - z_lr=1, steps=40 for NoOp
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'dcgan_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_64]
    },
    'dcgan': {
        # 'NoOp': [{}],  # Note - z_lr=1, steps=100 for NoOp
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'biggan': {
        # 'NoOp': [{}],  # Note - z_lr=1.5, steps=200 for NoOp
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'began_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'began_cs_other_init': {
        'GaussianCompressiveSensing': [{
            'n_measure': 8000
        }],
    },
    'began': {
        # 'NoOp': [{}],  # Note - z_lr=1, steps=100 for NoOp
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    }
}

baseline_settings = {
    'lasso-dct-64': {
        'img_size': 64,
        'lasso_coeff': [0.01] * len(n_measure_64),
        'n_measure': n_measure_64,
    },
    'lasso-dct-128': {
        'img_size': 128,
        'lasso_coeff': [0.01] * len(n_measure_128),
        'n_measure': n_measure_128,
    },
}

recovery_settings = {
    'began_opt_error_fake_imgs': {
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'optimizer': 'lbfgs',
        'n_steps': 100,
        'z_lr': 1,
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3],
    },
    'began_cs_other_init': {
        'z_init_mode':
        ['lasso_inverse', 'clamped_normal', 'normal', 'normal', 'zero'],
        'limit': [1, 1, 5, 10, None],
        'optimizer':
        'lbfgs',
        'n_steps':
        25,
        'z_lr':
        1,
        'recover_batch_size':
        1,
        'n_cuts_list': [1, 2, 3],
    },
    'began': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 5, 7, 9, 11, 13],
        'limit': [1],
    },
    'began_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 4, 5, 7, 9, 11, 13],
        'limit': [1],
    },
    'biggan': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1.5,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 7],
    },
    'dcgan': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 4],
        'limit': [1],
    },
    'dcgan_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 0.1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 4],
        'limit': [1],
    },
    'vanilla_vae': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 4, 5],
        'limit': [1],
    },
    'vanilla_vae_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 4, 5],
        'limit': [1],
    },
    'beta_vae': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 4, 5],
        'limit': [1],
    },
    'beta_vae_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'recover_batch_size': 1,
        'n_cuts_list': [0, 1, 2, 3, 4, 5],
        'limit': [1],
    },
    'iagan_began_cs': {
        'optimizer': 'adam',
        'z_steps1': 1600,
        'z_steps2': 600,
        'z_lr1': 1e-4,
        'z_lr2': 1e-4,
        'model_lr': 1e-3,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
    }
}
