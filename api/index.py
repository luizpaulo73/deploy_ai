from flask import Flask, request, jsonify
import joblib  
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://calculadoraverdi.streamlit.app/"])

# Carregar o modelo
modelo_pipeline = joblib.load('random_forest_pipeline_fixed.pkl')


@app.route('/')
def home():
    return "Bem-vindo à API de previsão!"

@app.route('/prever', methods=['POST'])
def prever():
    try:
        dados = request.get_json()  
        print("Dados recebidos:", dados)
        
        # Converter para DataFrame
        input_dados = pd.DataFrame([dados])

        categorical_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
        for cat_feature in categorical_features:
            categories = modelo_pipeline.named_steps['preprocessor'].transformers_[1][1].categories_[categorical_features.index(cat_feature)]
            print(f"Categories for {cat_feature}: {categories}")
            # Extract the 'Make' category from input_dados
            input_category = input_dados[cat_feature][0]  # Assuming you have a single data point in input_dados
            print(f"Input category for {cat_feature}: {input_category}")
            # Check if the input category is in the learned categories

        if input_category not in categories:
            print(f"WARNING: Input category '{input_category}' for {cat_feature} not found in learned categories.")
            
        # Ajustar colunas categóricas para os formatos esperados pelo pipeline
        for coluna in ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]:
            if coluna in input_dados.columns:
                input_dados[coluna] = input_dados[coluna].astype(str)

        print("X_train shape:", modelo_pipeline.named_steps['preprocessor'].transformers_[1][1].categories_)  # Print the shape of X_train
        print("input_dados shape:", input_dados.shape)  # Print the shape of input_dados

        # Realizar a predição
        prediction = modelo_pipeline.predict(input_dados)
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
