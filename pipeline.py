import os
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterBoolean,
    ParameterString
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator

def create_pipeline(role, bucket_name):
    input_size = ParameterInteger(name="input_size", default_value=224)
    random_resize_range = ParameterFloat(name="random_resize_range", default_value=0.5)
    datasets_names = ParameterString(name="datasets_names", default_value="chexpert")
    
    batch_size = ParameterInteger(name="batch_size", default_value=6)
    model = ParameterString(name="model", default_value="densenet121")
    mask_ratio = ParameterFloat(name="mask_ratio", default_value=0.75)
    epochs = ParameterInteger(name="epochs", default_value=11)
    warmup_epochs = ParameterInteger(name="warmup_epochs", default_value=5)
    blr = ParameterFloat(name="blr", default_value=1.5e-4)
    weight_decay = ParameterFloat(name="weight_decay", default_value=0.05)
    num_workers = ParameterInteger(name="num_workers", default_value=11)
    
    processor = ScriptProcessor(
        role=role,
        command=['python3'],
        instance_count=1,
        instance_type='ml.c5.xlarge',
    )
    
    processing_step = step(processor.run)(
        code='processing.py',
        inputs=[],
        outputs=[],
        arguments=[
            '--input_size', input_size,
            '--random_resize_range', random_resize_range,
            '--datasets_names', datasets_names
        ]
    )

    estimator = Estimator(
        role=role,
        instance_count=1,
        instance_type='ml.c5.xlarge',
        output_path=f's3://{bucket_name}/model',
    )
    
    training_step = step(estimator.fit)(
        inputs=processing_step,
        arguments=[
            '--output_dir', './OUTPUT_densenet121/',
            '--log_dir', './LOG_densenet121/',
            '--batch_size', batch_size,
            '--model', model,
            '--mask_ratio', mask_ratio,
            '--epochs', epochs,
            '--warmup_epochs', warmup_epochs,
            '--blr', blr,
            '--weight_decay', weight_decay,
            '--num_workers', num_workers,
            '--input_size', input_size,
            '--random_resize_range', random_resize_range,
            '--datasets_names', datasets_names,
            '--device', 'cpu'
        ]
    )

    evaluation_processor = ScriptProcessor(
        role=role,
        command=['python3'],
        instance_count=1,
        instance_type='ml.c5.xlarge',
    )
    
    evaluate_step = step(evaluation_processor.run)(
        code='evaluate.py',
        inputs=[training_step],
        arguments=[
            '--batch_size', batch_size,
            '--finetune', './OUTPUT_densenet121/Pretrain_densenet121.pth',
            '--model', model,
            '--data_path', 'data/CheXpert-v1.0/',
            '--num_workers', 2,
            '--train_list', 'data/CheXpert-v1.0/train.csv',
            '--val_list', 'data/CheXpert-v1.0/test1.csv',
            '--test_list', 'data/CheXpert-v1.0/test1.csv',
            '--nb_classes', 5,
            '--eval_interval', 10,
            '--dataset', datasets_names,
            '--aa', 'rand-m6-mstd0.5-inc1',
            '--device', 'cpu'
        ]
    )

    register_step = step(register)(
        role=role,
        inputs=[evaluate_step],
        model_package_group_name='model-group', 
    )

    deploy_step = step(deploy)(
        role=role,
        model_package_arn=register_step  
    )

    pipeline = Pipeline(
        name='MyPipeline',
        parameters=[
            input_size,
            random_resize_range,
            datasets_names,
            batch_size,
            model,
            mask_ratio,
            epochs,
            warmup_epochs,
            blr,
            weight_decay,
            num_workers
        ],
        steps=[processing_step, training_step, evaluate_step, register_step, deploy_step]
    )

    return pipeline

if __name__ == "__main__":
    role = get_execution_role()
    bucket_name = Session().default_bucket()

    pipeline = create_pipeline(role, bucket_name)

    pipeline.upsert(role_arn=role)
    pipeline.start()