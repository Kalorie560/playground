"""
Streamlit Web Application for Spaceship Titanic Prediction
宇宙船タイタニック予測のためのStreamlitウェブアプリケーション
"""

import streamlit as st
import torch
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

# Import our modules
from data_preprocessing import SpaceshipDataProcessor
from model import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="宇宙船タイタニック予測",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .prediction-result {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .transported {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .not-transported {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .feature-help {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

class SpaceshipWebPredictor:
    """Web predictor class for Streamlit app."""
    
    def __init__(self):
        self.config = self.load_config()
        self.device = torch.device('cpu')  # Use CPU for web app
        self.model = None
        self.data_processor = SpaceshipDataProcessor()
        self.feature_names = None
        self._init_model()
    
    def load_config(self):
        """Load configuration."""
        try:
            with open('config.yaml', 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                'model': {
                    'hidden_sizes': [256, 128, 64],
                    'dropout_rates': [0.3, 0.4, 0.5],
                    'activation': 'relu',
                    'use_batch_norm': True,
                    'output_size': 1
                }
            }
    
    @st.cache_resource
    def _init_model(_self):
        """Initialize model with dummy data to get feature structure."""
        # Create sample data to initialize the processor
        _self.data_processor = SpaceshipDataProcessor()
        train_df, test_df = _self.data_processor._create_sample_data()
        
        # Process sample data to get feature names
        processed_data = _self.data_processor.preprocess(train_df, test_df)
        _self.feature_names = processed_data['feature_names']
        input_size = len(_self.feature_names)
        
        # Create model
        _self.model = create_model(input_size, _self.config)
        _self.model.eval()
        
        logger.info(f"Model initialized with {input_size} features")
        return _self.model, _self.feature_names
    
    def create_sample_input(self, user_inputs):
        """Create a sample input DataFrame from user inputs."""
        # Create a base sample similar to the training data
        sample_data = {
            'PassengerId': '0001_01',
            'HomePlanet': user_inputs['home_planet'],
            'CryoSleep': user_inputs['cryo_sleep'],
            'Cabin': f"{user_inputs['deck']}/{user_inputs['cabin_num']}/{user_inputs['side']}",
            'Destination': user_inputs['destination'],
            'Age': user_inputs['age'],
            'VIP': user_inputs['vip'],
            'RoomService': user_inputs['room_service'],
            'FoodCourt': user_inputs['food_court'],
            'ShoppingMall': user_inputs['shopping_mall'],
            'Spa': user_inputs['spa'],
            'VRDeck': user_inputs['vr_deck'],
            'Name': 'User Input'
        }
        
        return pd.DataFrame([sample_data])
    
    def predict(self, user_inputs):
        """Make prediction from user inputs."""
        # Create sample input DataFrame
        input_df = self.create_sample_input(user_inputs)
        
        # Create a dummy training set for the processor
        train_df, _ = self.data_processor._create_sample_data()
        
        # Preprocess the input
        processed_data = self.data_processor.preprocess(train_df, input_df)
        X_input = processed_data['X_test']
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_input.values)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(X_tensor)
            probability = torch.sigmoid(output).item()
        
        return probability

def main():
    """Main Streamlit app function."""
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        with st.spinner('モデルを初期化中...'):
            st.session_state.predictor = SpaceshipWebPredictor()
    
    predictor = st.session_state.predictor
    
    # Header
    st.markdown('<h1 class="main-header">🚀 宇宙船タイタニック予測システム</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">あなたが別次元に輸送されるかどうかを予測します</h3>', unsafe_allow_html=True)
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 使用方法")
        st.markdown("""
        1. 左側のフォームに個人情報を入力してください
        2. 各項目には適切な値を入力してください
        3. 「予測する」ボタンをクリックします
        4. AI モデルが輸送確率を計算します
        """)
        
        st.header("ℹ️ 特徴量について")
        st.markdown("""
        - **ホームプラネット**: 出身惑星
        - **冷凍睡眠**: 航行中の冷凍睡眠状態
        - **客室**: デッキ/番号/サイド（Port/Starboard）
        - **目的地**: 宇宙船の最終目的地
        - **年齢**: 乗客の年齢
        - **VIP**: VIP ステータス
        - **サービス料金**: 各種船内サービスの利用料金
        """)
    
    # Main input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧑‍🚀 個人情報")
        
        home_planet = st.selectbox(
            "ホームプラネット (Home Planet)",
            options=['Europa', 'Earth', 'Mars'],
            help="出身惑星を選択してください"
        )
        
        age = st.slider(
            "年齢 (Age)",
            min_value=0,
            max_value=100,
            value=30,
            help="年齢を入力してください（0-100歳）"
        )
        
        vip = st.checkbox(
            "VIP ステータス",
            help="VIP サービスを利用していますか？"
        )
        
        cryo_sleep = st.checkbox(
            "冷凍睡眠 (CryoSleep)",
            help="航行中に冷凍睡眠状態でしたか？"
        )
        
        destination = st.selectbox(
            "目的地 (Destination)",
            options=['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'],
            help="宇宙船の最終目的地を選択してください"
        )
    
    with col2:
        st.subheader("🛏️ 客室情報")
        
        deck = st.selectbox(
            "デッキ (Deck)",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            help="客室のデッキを選択してください"
        )
        
        cabin_num = st.number_input(
            "客室番号 (Cabin Number)",
            min_value=1,
            max_value=1999,
            value=100,
            help="客室番号を入力してください"
        )
        
        side = st.selectbox(
            "サイド (Side)",
            options=['P', 'S'],
            help="Port（左舷）またはStarboard（右舷）"
        )
        
        st.subheader("💰 サービス料金 (Credits)")
        
        room_service = st.number_input(
            "ルームサービス",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=10.0,
            help="ルームサービスの利用料金"
        )
        
        food_court = st.number_input(
            "フードコート",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=10.0,
            help="フードコートでの支出額"
        )
        
        shopping_mall = st.number_input(
            "ショッピングモール",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=10.0,
            help="ショッピングモールでの支出額"
        )
        
        spa = st.number_input(
            "スパ",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=10.0,
            help="スパサービスの利用料金"
        )
        
        vr_deck = st.number_input(
            "VRデッキ",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=10.0,
            help="VRデッキの利用料金"
        )
    
    # Prediction button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 予測する",
            use_container_width=True,
            type="primary"
        )
    
    # Make prediction
    if predict_button:
        # Collect user inputs
        user_inputs = {
            'home_planet': home_planet,
            'age': age,
            'vip': vip,
            'cryo_sleep': cryo_sleep,
            'destination': destination,
            'deck': deck,
            'cabin_num': cabin_num,
            'side': side,
            'room_service': room_service,
            'food_court': food_court,
            'shopping_mall': shopping_mall,
            'spa': spa,
            'vr_deck': vr_deck
        }
        
        try:
            with st.spinner('予測を計算中...'):
                probability = predictor.predict(user_inputs)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 予測結果")
            
            if probability > 0.5:
                st.markdown(f"""
                <div class="prediction-result transported">
                    ✅ 輸送される可能性が高いです<br>
                    確率: {probability:.1%}
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-result not-transported">
                    ❌ 輸送されない可能性が高いです<br>
                    確率: {probability:.1%}
                </div>
                """, unsafe_allow_html=True)
            
            # Additional information
            st.markdown("### 📊 詳細情報")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("輸送確率", f"{probability:.1%}")
            with col2:
                st.metric("非輸送確率", f"{1-probability:.1%}")
            with col3:
                total_spending = room_service + food_court + shopping_mall + spa + vr_deck
                st.metric("総支出額", f"{total_spending:,.0f} Credits")
            
            # Show input summary
            with st.expander("入力データの確認"):
                input_df = pd.DataFrame([user_inputs])
                st.dataframe(input_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"予測中にエラーが発生しました: {str(e)}")
            logger.error(f"Prediction error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
        🤖 このアプリケーションは深層学習モデルを使用しています<br>
        Built with Streamlit & PyTorch | Spaceship Titanic Prediction System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()