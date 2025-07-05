# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
from io import BytesIO
import packaging.version

# 设置中文字体支持
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 自定义 CSS 样式（新增shap-title类）
custom_css = """
<style>
    h1 { color: #007BFF; text-align: center; font-size: 28px; margin-bottom: 20px; }
    h2 { color: #333; font-size: 18px; text-align: center; margin-bottom: 15px; }
    .shap-title {  /* 新增样式类 */
        font-size: 16px;  /* 进一步减小字体 */
        white-space: nowrap;  /* 防止文本换行 */
        overflow: hidden;  /* 隐藏溢出内容 */
        text-overflow: ellipsis;  /* 溢出内容用省略号表示 */
        margin-bottom: 10px;  /* 调整间距 */
    }
    .main .block-container { padding-top: 1rem; }
    .css-12oz5g7 { padding: 0 1rem; }
    .stButton>button { background-color: #28A745; color: white; border-radius: 5px; padding: 10px 20px; font-weight: bold; width: 100%; margin-top: 15px; }
    .stSuccess { background-color: #D4EDDA; color: #155724; padding: 12px; border-radius: 5px; font-weight: 500; text-align: center; }
    .stError { background-color: #F8D7DA; color: #721C24; padding: 12px; border-radius: 5px; font-weight: 500; text-align: center; }
    .shap-plot-container { margin-top: 20px; border-radius: 5px; background-color: #f8f9fa; padding: 15px; }
    .stTextInput>div>div>input, .stSelectbox>div>div>select { border-radius: 5px; border: 1px solid #CED4DA; padding: 5px; }
    .shap-svg-container {
        transform: scale(0.5);  /* 调整为50%缩放 */
        transform-origin: top left;
        display: inline-block;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 数据加载与预处理函数
@st.cache_data
def load_and_preprocess_data(train_path, test_path):
    try:
        # 加载数据
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # 定义变量类型
        continuous_vars = ["SWS", "A", "ADC", "T1_Mapping1", "T1_Mapping2"]
        categorical_vars = {
            "High_signal_intensity_on_T2WI": [0, 1],
            "Perienteric_fat_stranding_on_T2WI": [0, 1],
            "Length_of_the_affected_bowel_segment_on_T1WI": [1, 2],
            "Bowel_stricture_on_T1WI": [0, 1, 2],
            "Penetrating_lesion_on_T1WI": [0, 1, 2],
            "Perianal_lesion_on_T1WI": [0, 1, 2, 3]
        }
        
        # 训练集连续型变量标准化
        scaler = StandardScaler()
        train_data[continuous_vars] = scaler.fit_transform(train_data[continuous_vars])
        test_data[continuous_vars] = scaler.transform(test_data[continuous_vars])
        
        # 训练集因变量Fibrosis_score因子化
        train_data["Fibrosis_score"] = train_data["Fibrosis_score"].astype('category')
        test_data["Fibrosis_score"] = test_data["Fibrosis_score"].astype('category')
        
        # 定义训练集特征，分类变量转换为哑变量
        train_x = pd.get_dummies(train_data.drop("Fibrosis_score", axis=1))
        train_y = train_data["Fibrosis_score"].cat.codes  # 标签转换为数值向量
        
        # 定义验证集特征，分类变量转换为哑变量
        val_x = pd.get_dummies(test_data.drop("Fibrosis_score", axis=1))
        val_y = test_data["Fibrosis_score"].cat.codes  # 标签转换为数值向量
        
        # 确保训练集和测试集特征一致
        missing_cols = set(train_x.columns) - set(val_x.columns)
        for col in missing_cols:
            val_x[col] = 0
        val_x = val_x[train_x.columns]
        
        # 保存特征名称和类别映射
        feature_names = train_x.columns.tolist()
        class_mapping = {
            0: "None-to-mild Fibrosis",
            1: "Moderate-to-severe Fibrosis"
        }
        
        return train_x, train_y, val_x, val_y, feature_names, class_mapping, scaler, continuous_vars, categorical_vars
    
    except Exception as e:
        st.error(f"Data loading or preprocessing error: {e}")
        st.stop()

# 模型训练函数
@st.cache_resource
def train_xgboost_model(train_x, train_y):
    # 使用xgb.DMatrix函数将特征和目标变量转换为DMatrix矩阵格式
    dtrain = xgb.DMatrix(data=train_x, label=train_y)
    
    # 设置最佳XGBoost参数，添加类别权重
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.01,
        "gamma": 0.5,
        "colsample_bytree": 1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "seed": 316,
        # 调整类别权重，缓解类别不平衡问题
        "scale_pos_weight": float(np.sum(train_y == 0)) / np.sum(train_y == 1)
    }
    
    # 拟合模型
    xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)
    
    return xgb_model

# 定义特征名称映射字典
feature_display_mapping = {
    "SWS": "SWS",
    "A": "φ",
    "ADC": "ADC",
    "T1_Mapping1": "Pre-contrast T1 value",
    "T1_Mapping2": "Post-contrast T1 value",
    "High_signal_intensity_on_T2WI": "Mural T2WI hyperintensity",
    "Perienteric_fat_stranding_on_T2WI": "Creeping fat",
    "Length_of_the_affected_bowel_segment_on_T1WI": "Long-segment lesion",
    "Bowel_stricture_on_T1WI": "Stenosis",
    "Penetrating_lesion_on_T1WI": "Penetrating lesion",
    "Perianal_lesion_on_T1WI": "Perianal lesion"
}

# 生成SHAP瀑布图并转换为SVG
def generate_shap_waterfall_plot_svg(explainer, shap_values, features, original_feature_names, sample_idx=0):
    """生成SHAP瀑布图并转换为SVG格式，使用自定义特征名称"""
    try:
        # 创建特征名称映射，处理哑变量
        display_feature_names = []
        for feature in original_feature_names:
            # 处理连续变量
            if feature in feature_display_mapping:
                display_feature_names.append(feature_display_mapping[feature])
            # 处理分类变量的哑变量
            else:
                # 尝试从特征名中提取原始分类特征
                for cat_feature in feature_display_mapping:
                    if feature.startswith(cat_feature):
                        # 提取后缀值（如 _1, _2 等）
                        suffix = feature.split('_')[-1]
                        if suffix.isdigit():
                            display_name = f"{feature_display_mapping[cat_feature]}={suffix}"
                        else:
                            display_name = feature_display_mapping[cat_feature]
                        display_feature_names.append(display_name)
                        break
                else:
                    # 未找到匹配的分类特征，使用原始特征名
                    display_feature_names.append(feature)
        
        # 创建瀑布图
        plt.figure(figsize=(10, 6))
        
        # 处理二分类问题
        if isinstance(shap_values, list) and len(shap_values) == 2:
            base_value = explainer.expected_value[1]
            shap_values = shap_values[1]
        else:
            base_value = explainer.expected_value
        
        # 获取样本的预测值
        pred_value = base_value + shap_values[sample_idx].sum()
        
        # 检查SHAP版本
        shap_version = packaging.version.parse(shap.__version__)
        
        # 根据版本使用不同参数
        if shap_version >= packaging.version.parse("0.41.0"):
            # 新版本API
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=base_value,
                    data=features.iloc[sample_idx],
                    feature_names=display_feature_names  # 使用自定义显示名称
                ),
                max_display=10  # 减少显示的特征数量
            )
        else:
            # 旧版本API
            kwargs = {
                'max_display': 10  # 减少显示的特征数量
            }
            # 只有在版本支持时才添加show_values参数
            if shap_version >= packaging.version.parse("0.40.0"):
                kwargs['show_values'] = False
            
            shap.plots._waterfall.waterfall_legacy(
                base_value,
                shap_values[sample_idx],
                features.iloc[sample_idx],
                feature_names=display_feature_names,  # 使用自定义显示名称
                **kwargs
            )
        
        # 调整布局
        plt.tight_layout(pad=1.0)  # 减小边距
        
        # 保存为SVG
        svg_io = BytesIO()
        plt.savefig(svg_io, format='svg', bbox_inches='tight')
        plt.close()
        svg_io.seek(0)
        return svg_io.getvalue().decode('utf-8')
    
    except Exception as e:
        st.error(f"生成SHAP瀑布图失败: {e}")
        return None

# 主应用
def main():
    st.title("XGBoost Model for Intestinal Fibrosis Severity Assessment")
    
    # 直接指定数据路径（可根据需要修改）
    train_data_path = "lasso_data_train.csv"
    test_data_path = "lasso_data_test.csv"
    
    # 加载和预处理数据
    train_x, train_y, val_x, val_y, feature_names, class_mapping, scaler, continuous_vars, categorical_vars = load_and_preprocess_data(
        train_data_path, test_data_path
    )
    
    # 训练模型
    xgb_model = train_xgboost_model(train_x, train_y)
    
    # 三列布局
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # 初始化输入特征字典
    input_features = {}
    
    # 第一列：定量特征
    with col1:
        st.subheader("Quantitative Features")
        
        display_name_mapping = {
            "SWS": "SWS",
            "A": "φ",
            "ADC": "ADC",
            "T1_Mapping1": "Pre-contrast T1 value",
            "T1_Mapping2": "Post-contrast T1 value",
        }
        
        # 保存原始输入值，用于调试
        raw_inputs = {}
        
        # 处理连续变量
        for feature in continuous_vars:
            if feature in feature_names:
                display_name = display_name_mapping.get(feature, feature)
                default_value = 0.0
                raw_inputs[feature] = st.number_input(f"{display_name}", value=default_value)
    
    # 第二列：定性特征
    with col2:
        st.subheader("Categorical Features")
        
        display_name_mapping = {
            "High_signal_intensity_on_T2WI": "Mural T2WI hyperintensity",
            "Perienteric_fat_stranding_on_T2WI": "Creeping fat",
            "Length_of_the_affected_bowel_segment_on_T1WI": "Long-segment lesion",
            "Bowel_stricture_on_T1WI": "Stenosis",
            "Penetrating_lesion_on_T1WI": "Penetrating lesion",
            "Perianal_lesion_on_T1WI": "Perianal lesion"
        }
        
        option_descriptions = {
            "High_signal_intensity_on_T2WI": {0: "0: None", 1: "1: Yes"},
            "Perienteric_fat_stranding_on_T2WI": {0: "0: None", 1: "1: Yes"},
            "Length_of_the_affected_bowel_segment_on_T1WI": {1: "1: ≤15cm", 2: "2: >15cm"},
            "Bowel_stricture_on_T1WI": {0: "0: None", 1: "1: Stenosis not with prestenosis dilation", 2: "2: Stenosis with prestenosis dilation"},
            "Penetrating_lesion_on_T1WI": {0: "0: None", 1: "1: Anabrosis", 2: "2: Intestinal fistula/peri-intestinal abscess"},
            "Perianal_lesion_on_T1WI": {0: "0: None", 1: "1: Anal fistula", 2: "2: Perianal abscess", 3: "3: Anal fistula + perianal abscess"}
        }
        
        # 处理分类变量
        for cat_feature in categorical_vars:
            if cat_feature in categorical_vars:
                display_name = display_name_mapping.get(cat_feature, cat_feature)
                values = categorical_vars[cat_feature]
                
                dummy_features = [f for f in feature_names if f.startswith(cat_feature)]
                
                if not dummy_features:
                    st.warning(f"Dummy features not found for '{cat_feature}'")
                    continue
                
                possible_values = []
                for f in dummy_features:
                    try:
                        parts = f.split('_')
                        value_part = parts[-1]
                        value = int(value_part)
                        possible_values.append(value)
                    except (ValueError, IndexError):
                        possible_values = values
                        break
                
                possible_values = sorted(list(set(possible_values)))
                
                display_options = [option_descriptions[cat_feature].get(v, str(v)) for v in possible_values]
                display_to_value = {option_descriptions[cat_feature].get(v, str(v)): v for v in possible_values}
                
                selected_display = st.selectbox(
                    f"{display_name}",
                    display_options,
                    key=cat_feature
                )
                
                selected_value = display_to_value[selected_display]
                
                for dummy_feature in dummy_features:
                    try:
                        parts = dummy_feature.split('_')
                        dummy_value = int(parts[-1])
                    except (ValueError, IndexError):
                        dummy_value = 0
                    
                    input_features[dummy_feature] = 1 if dummy_value == selected_value else 0
        
        # 预测按钮
        predict_button = st.button("Predict")
    
    # 第三列：预测结果
    with col3:
        st.subheader("Prediction Results")
        
        if predict_button:
            try:
                # 对连续变量进行标准化
                continuous_input = pd.DataFrame([raw_inputs])
                scaled_continuous = scaler.transform(continuous_input[continuous_vars])
                
                # 将标准化后的连续变量添加到输入特征中
                for i, feature in enumerate(continuous_vars):
                    input_features[feature] = scaled_continuous[0, i]
                
                # 创建输入DataFrame
                input_df = pd.DataFrame([input_features])
                
                # 确保输入特征与训练时一致
                for col in train_x.columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[train_x.columns]
                
                # 模型预测
                dinput = xgb.DMatrix(data=input_df)
                prediction_proba = xgb_model.predict(dinput)[0]
                
                # 打印预测概率
                st.write(f"Prediction Probability: {prediction_proba:.4f}")
                
                # 使用更灵活的阈值（可根据实际情况调整）
                threshold = 0.5
                prediction_class = 1 if prediction_proba >= threshold else 0
                
                # 使用class_mapping获取描述性标签
                decoded_prediction = class_mapping[prediction_class]
                
                # 显示预测结果
                st.success(f"Prediction: {decoded_prediction}")
                
                # 初始化 SHAP 解释器
                try:
                    # 尝试使用最新API
                    explainer = shap.Explainer(xgb_model)
                    shap_values = explainer(input_df)
                    
                    # 处理不同版本的SHAP输出格式
                    if hasattr(shap_values, 'values'):
                        shap_values_array = shap_values.values
                        # 处理二分类问题
                        if len(shap_values_array.shape) == 3:
                            shap_values_array = shap_values_array[:, :, 1]  # 取正类的SHAP值
                    else:
                        shap_values_array = shap_values
                    
                except Exception as e:
                    # 回退到旧API
                    st.warning(f"Use the old version of SHAP API: {e}")
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values_array = explainer.shap_values(input_df)
                    
                    # 处理二分类问题
                    if isinstance(shap_values_array, list) and len(shap_values_array) == 2:
                        shap_values_array = shap_values_array[1]  # 取正类的SHAP值
                
                # 显示 SHAP 瀑布图
                if 'shap_values_array' in locals():
                    st.markdown('<h3 class="shap-title">SHAP Summary Plot</h3>', unsafe_allow_html=True)  # 使用新样式
                    # 生成SVG格式的SHAP瀑布图
                    svg_content = generate_shap_waterfall_plot_svg(explainer, shap_values_array, input_df, feature_names)
                    
                    if svg_content:
                        # 使用容器和CSS缩放SVG
                        st.markdown(f'<div class="shap-svg-container">{svg_content}</div>', unsafe_allow_html=True)
                    else:
                        st.error("The SHAP waterfall graph cannot be generated.")
                        # 显示文本SHAP值作为备选
                        st.subheader("SHAP Values (Text)")
                        shap_df = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP Value': shap_values_array[0]
                        })
                        st.dataframe(shap_df.sort_values('SHAP Value', ascending=False))
            
            except Exception as e:
                st.error(f"An error occurred during the prediction process: {e}")
        else:
            st.info("Please enter all the features and click'Predict'")

if __name__ == "__main__":
    main()
