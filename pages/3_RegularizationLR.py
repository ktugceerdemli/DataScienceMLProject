import streamlit as st
from sklearn.datasets import make_regression
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
######### Styling ##############

st.markdown("""
    <style>
    .custom-font-head {
        font-size:20px;
        font-weight:bold;
        font-family:'Arial';
    }
    .custom-font-head-mid-small {
        font-size:30px;
        font-weight:bold;
        font-family:'Arial';
    }
    .custom-font-write {
        font-size:18px;
        font-weight:italic;
        font-family:'Arial';
    }
    .custom-font-head-mid {
        font-size:50px;
        font-weight:italic;
        font-family:'Arial';
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""<div style="text-align: center; margin: 20px">
            <p class="custom-font-head-mid"> Regularizasyon </p> </div>""", unsafe_allow_html=True)

st.markdown("""<p class="custom-font-write"> Varyans'ın yüksek olduğu durumlarda aşırı uyum (overfitting) oluşabilir ya da model çok karmaşıktır ve bunu düzenlemek gerekebilir.   
                Ridge Regülasyonu, Lasso Regülasyonu, Elastic-Net Regülasyonu Doğrusal regresyon için kullanılan regularizasyon yöntemlerindendir.
                </p>""", unsafe_allow_html=True)

st.markdown("""<p class="custom-font-write"> Bu modeller, özellikle yüksek boyutlu verilerle çalışırken modelin performansını iyileştirmek için kullanılır.
                Regularizasyonlar da genel olarak, modelin kayıp fonksiyonuna (cost function ) eklenen ceza terimleri eklenerek elde edilir. Böylece modelin ağırlıkları düzenlenerek daha dengeli ve genellenebilir bir denklem oluşturulur.
                </p>""", unsafe_allow_html=True)

st.markdown('<p class="custom-font-head">Veri Seti Oluşturma</p>', unsafe_allow_html=True)

st.markdown('<p class="custom-font-write">Oluşturulacak veri seti büyüklüğünü giriniz.</p>', unsafe_allow_html=True)

num_samples = st.slider("Veri Seti Büyüklüğü", 2, 100, 50)

st.markdown('<p class="custom-font-write">Oluşturulacak gürültüyü giriniz.</p>', unsafe_allow_html=True)

noise = st.slider("Gürültü ekleme", 0, 100, 10)

# make_regression kullanılarak veri seti oluşturuldu. / Prepare dataset with make_regression library
X, y = make_regression(n_samples=num_samples, n_features=2, noise=noise, random_state=43)

st.markdown('<p class="custom-font-head"> Veri setini inceleyelim </p>', unsafe_allow_html=True)
# Veri setini DataFrame olarak düzenleme
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['y'] = y

col1, col2 = st.columns([3, 7])
with col1:
    st.markdown('<p class="custom-font-write">Veri setimiz: </p>', unsafe_allow_html=True)
    st.dataframe(df, height=350)

with col2:
    st.markdown('<p class="custom-font-write">3 boyutlu Dağılım grafiği: </p>', unsafe_allow_html=True)
    fig = px.scatter_3d(df, x='X1', y='X2', z='y')
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig)

tab1, tab2, tab3, tab4 = st.tabs(['Ridge', 'Lasso', 'Elastic-Net','Ridge-Linear'])
###### Ridge Regression ################
with tab1:
    st.markdown('<p class="custom-font-head-mid-small"> Ridge Regression </p>', unsafe_allow_html=True)

    st.markdown('<p class="custom-font-write"> Ridge regresyon, modelin katsayılarını küçültmek için ceza terimi olarak katsayıların karelerinin toplamını ekler. Bu ceza terimi, modelin ağırlığının büyüklüğünü kısıtlar böylece modelin karmaşıklığını kontrol eder ve overfitting\'i önelemeye yardımcı olur. </p>', unsafe_allow_html=True)

    st.markdown('<p class="custom-font-write"> Veri setimizin çok yüksek olduğu durumlarda modelin overfitting olmasını engeller. </p>',
        unsafe_allow_html=True)

    formul1 = r"""
    $$
    J(w) = \sum_{i=1}^{n} \left( y_i - w^{\top} x_i \right)^2 + \lambda \sum_{j=1}^{p} w_j^2
    $$
    """

    formul2 = r"""
    $$
    \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X} + \lambda I)^{-1} \mathbf{X}^T \mathbf{y}
    $$
    """

    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px;">
            <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
                <p style="font-size: 24px; font-weight: bold;">
                     {formul1} </p></div></div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 20px;">
                <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
                    <p style="font-size: 24px; font-weight: bold;">
                         {formul2} </p></div></div>
            """, unsafe_allow_html=True)

    code = '''
    def ridge_regression(X, y, alpha):
    X_T = X.T  
    I = np.eye(X.shape[1])  # Birim matris
    I[0, 0] = 0  # Bias terimine ceza eklemiyoruz

    # Ridge regresyon katsayıları hesaplama
    beta = np.linalg.inv(X_T @ X + alpha * I) @ X_T @ y
    return beta

    '''

    st.code(code, language='python')
    # Bias terimi ekleme
    df.insert(0, 'Intercept', 1)

    # Veriyi eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(df[['Intercept', 'X1', 'X2']], df['y'], test_size=0.3, random_state=42)

    # NumPy matrisleri oluşturma
    X_matrix_train = X_train.values
    X_matrix_test = X_test.values
    y_vector_train = y_train.values.reshape(-1, 1)
    y_vector_test = y_test.values.reshape(-1, 1)

    # Ridge modeli tanımlama
    st.header("Ridge Parametreleri")
    alpha = st.slider("Lambda (α) Değeri:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)


    # Ridge regresyonunu manuel olarak uygulama
    def ridge_regression(X, y, alpha):
        X_T = X.T  # X matrisinin transpozesi
        I = np.eye(X.shape[1])  # Birim matris
        I[0, 0] = 0  # Bias terimine ceza eklemiyoruz

        # Ridge regresyon katsayıları hesaplama
        beta = np.linalg.inv(X_T @ X + alpha * I) @ X_T @ y
        return beta

    # Manuel Ridge katsayılarını hesaplama
    ridge_coef_manual = ridge_regression(X_matrix_train, y_vector_train, alpha)

    # Test setinde tahmin
    y_pred_manual = X_matrix_test @ ridge_coef_manual
    mse_manual = mean_squared_error(y_vector_test, y_pred_manual)

    st.write(f"**Manuel Ridge Regression - Test Setinde Hata (MSE):** {mse_manual:.4f}")

    # Ridge Katsayıları Görüntüleme (Manuel)
    features = ['Intercept', 'X1', 'X2']  # Bias terimi ile birlikte
    st.write("**Manuel Ridge Regresyon Katsayıları:**")
    st.write(pd.DataFrame(ridge_coef_manual, index=features, columns=["Katsayı"]))

    st.markdown('<p class="custom-font-head"> Manuel Ridge Regression formülü </p>',
                unsafe_allow_html=True)
    st.markdown(
        f"""
            <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 5px; margin: 10px;">
            <p style="font-size: 24px; font-weight: bold;">y = {ridge_coef_manual[1][0]:.2f}x1 + {ridge_coef_manual[2][0]:.2f}x2 + {ridge_coef_manual[0][0]:.2f}</p>
        </div>
            """,
        unsafe_allow_html=True
    )
    # sklearn kullanarak Ridge regresyonu
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(df[['X1', 'X2']], df['y'], test_size=0.3,
                                                                    random_state=42)

    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_sk, y_train_sk)

    # Test setinde tahmin (sklearn)
    y_pred_sklearn = ridge_model.predict(X_test_sk)
    mse_sklearn = mean_squared_error(y_test_sk, y_pred_sklearn)

    st.write(f"**Sklearn Ridge Regression - Test Setinde Hata (MSE):** {mse_sklearn:.4f}")

    # Ridge Katsayıları Görüntüleme (Sklearn)
    # Bias terimini manuel olarak ekliyoruz
    ridge_coef_sklearn = np.concatenate([[ridge_model.intercept_], ridge_model.coef_])
    st.write("**Sklearn Ridge Regresyon Katsayıları:**")
    st.write(pd.DataFrame(ridge_coef_sklearn, index=['Intercept'] + ['X1', 'X2'], columns=["Katsayı"]))

    st.markdown('<p class="custom-font-head"> Sckit-learn kütüphanesinde ki Ridge Regression formülü </p>',
                unsafe_allow_html=True)
    st.markdown(
        f"""
                <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 5px; margin: 10px;">
                <p style="font-size: 24px; font-weight: bold;">y = {ridge_coef_sklearn[1]:.2f}x1 + {ridge_coef_sklearn[2]:.2f}x2 + {ridge_coef_sklearn[0]:.2f}</p>
            </div>
                """,
        unsafe_allow_html=True
    )


    ###### Lasso Regression ################
with tab2:
    st.markdown('<p class="custom-font-head-mid-small"> Lasso Regression </p>', unsafe_allow_html=True)

    st.markdown(
        '<p class="custom-font-write"> Lasso regresyon, modelin katsayılarını küçültmek için mutlak değer ceza terimi ekler. Bu ceza terimi, modelin karmaşıklığını kontrol etmeye yardımcı olur. Modelin bazı katsayılarını sıfıra eşitleyerek önemli değişkenleri seçmeye yardımcı olur. </p>',
        unsafe_allow_html=True)

    st.markdown(
        '<p class="custom-font-write"> Özellikle değişken seçimi ve modelin basitleştirilmesi gerektiğinde kullanılır. Verinin çok fazla özelliği varsa, gereksiz olanları otomatik olarak seçip atar. </p>',
        unsafe_allow_html=True)

    formul1 = r"""
    $$
    J(w) = \sum_{i=1}^{n} \left( y_i - w^{\top} x_i \right)^2 + \lambda \sum_{j=1}^{p} |w_j|
    $$
    """

    formul2 = r"""
    $$
    \boldsymbol{\beta} = \arg\min_{\mathbf{w}} \left\{ \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^{p} |\mathbf{w}_j| \right\}
    $$
    """

    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px;">
            <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
                <p style="font-size: 24px; font-weight: bold;">
                     {formul1} </p></div></div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 20px;">
                <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
                    <p style="font-size: 24px; font-weight: bold;">
                         {formul2} </p></div></div>
            """, unsafe_allow_html=True)

    code = '''
    def lasso_regression(X, y, alpha):
        X_T = X.T  
        I = np.eye(X.shape[1])  # Birim matris
        I[0, 0] = 0  # Bias terimine ceza eklemiyoruz

        # Lasso regresyon katsayıları hesaplama
        beta = np.linalg.inv(X_T @ X + alpha * I) @ X_T @ y
        return beta
    '''

    st.code(code, language='python')


    # Lasso modeli tanımlama
    st.header("Lasso Parametreleri")
    alpha_l = st.slider("Lambda (α) Değeri:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)


    # Lasso regresyonunu manuel olarak uygulama
    def lasso_regression(X, y, alpha):
        X_T = X.T  # X matrisinin transpozesi
        I = np.eye(X.shape[1])  # Birim matris
        I[0, 0] = 0  # Bias terimine ceza eklemiyoruz

        # Lasso regresyon katsayıları hesaplama
        beta = np.linalg.inv(X_T @ X + alpha * I) @ X_T @ y
        return beta


    # Manuel Lasso katsayılarını hesaplama
    lasso_coef_manual = lasso_regression(X_matrix_train, y_vector_train, alpha_l)

    # Test setinde tahmin
    y_pred_manual = X_matrix_test @ lasso_coef_manual
    mse_manual = mean_squared_error(y_vector_test, y_pred_manual)

    st.write(f"**Manuel Lasso Regression - Test Setinde Hata (MSE):** {mse_manual:.4f}")

    # Lasso Katsayıları Görüntüleme (Manuel)
    features = ['Intercept', 'X1', 'X2']  # Bias terimi ile birlikte
    st.write("**Manuel Lasso Regresyon Katsayıları:**")
    st.write(pd.DataFrame(lasso_coef_manual, index=features, columns=["Katsayı"]))

    st.markdown('<p class="custom-font-head"> Manuel Lasso Regression formülü </p>', unsafe_allow_html=True)
    st.markdown(
        f"""
            <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 5px; margin: 10px;">
            <p style="font-size: 24px; font-weight: bold;">y = {lasso_coef_manual[1][0]:.2f}x1 + {lasso_coef_manual[2][0]:.2f}x2 + {lasso_coef_manual[0][0]:.2f}</p>
        </div>
            """,
        unsafe_allow_html=True
    )

    # sklearn kullanarak Lasso regresyonu
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(df[['X1', 'X2']], df['y'], test_size=0.3,
                                                                    random_state=42)

    lasso_model = Lasso(alpha=alpha_l)
    lasso_model.fit(X_train_sk, y_train_sk)

    # Test setinde tahmin (sklearn)
    y_pred_sklearn = lasso_model.predict(X_test_sk)
    mse_sklearn = mean_squared_error(y_test_sk, y_pred_sklearn)

    st.write(f"**Sklearn Lasso Regression - Test Setinde Hata (MSE):** {mse_sklearn:.4f}")

    # Lasso Katsayıları Görüntüleme (Sklearn)
    # Bias terimini manuel olarak ekliyoruz
    lasso_coef_sklearn = np.concatenate([[lasso_model.intercept_], lasso_model.coef_])
    st.write("**Sklearn Lasso Regresyon Katsayıları:**")
    st.write(pd.DataFrame(lasso_coef_sklearn, index=['Intercept'] + ['X1', 'X2'], columns=["Katsayı"]))

    st.markdown('<p class="custom-font-head"> Sckit-learn kütüphanesinde ki Lasso Regression formülü </p>',
                unsafe_allow_html=True)
    st.markdown(
        f"""
                <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 5px; margin: 10px;">
                <p style="font-size: 24px; font-weight: bold;">y = {lasso_coef_sklearn[1]:.2f}x1 + {lasso_coef_sklearn[2]:.2f}x2 + {lasso_coef_sklearn[0]:.2f}</p>
            </div>
                """,
        unsafe_allow_html=True
    )

    ###### Elastic-Net Regression ################
with tab3:

    st.markdown('<p class="custom-font-head-mid-small"> Elastic-Net Regression </p>', unsafe_allow_html=True)

    st.markdown('<p class="custom-font-write"> Elastic-Net regresyon, Ridge ve Lasso regresyonun birleşimidir. Bu model, her iki yaklaşımın avantajlarını birleştirir.</p>', unsafe_allow_html=True)

    formul1 = r"""
    $$
    J(w) = \sum_{i=1}^{n} \left( y_i - w^{\top} x_i \right)^2 + \lambda \left[ (1-\alpha) \sum_{j=1}^{p} w_j^2 + \alpha \sum_{j=1}^{p} |w_j| \right]
    $$
    """

    formul2 = r"""
    $$
    \boldsymbol{\beta} = \arg\min_{\mathbf{w}} \left\{ \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \left[ (1-\alpha) \sum_{j=1}^{p} \mathbf{w}_j^2 + \alpha \sum_{j=1}^{p} |\mathbf{w}_j| \right] \right\}
    $$
    """

    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px;">
            <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
                <p style="font-size: 24px; font-weight: bold;">
                     {formul1} </p></div></div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 20px;">
                <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
                    <p style="font-size: 24px; font-weight: bold;">
                         {formul2} </p></div></div>
            """, unsafe_allow_html=True)


with tab4:
    st.title("Ridge, Lasso ve Linear Regresyon Karşılaştırması")

    # Regresyon Parametreleri
    data = st.slider("Veri sayısı:", min_value=2, max_value=1000, value=100)
    alpha_ = st.slider("Lambda (α) Değeri:", min_value=0.0, max_value=100.0, value=1.0, step=0.1)


    # Veri seti oluşturma (X1 ve y1)
    X1, y1 = make_regression(n_samples=data, n_features=1, noise=20, random_state=43)

    # Eğitim ve test setine ayırma
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

    # Linear Regression modeli
    linear_model = LinearRegression()
    linear_model.fit(X1_train, y1_train)
    y1_pred_linear = linear_model.predict(X1_test)
    mse_linear = mean_squared_error(y1_test, y1_pred_linear)

    # Ridge Regression modeli
    ridge_model = Ridge(alpha=alpha_)
    ridge_model.fit(X1_train, y1_train)
    y1_pred_ridge = ridge_model.predict(X1_test)
    mse_ridge = mean_squared_error(y1_test, y1_pred_ridge)

    # Lasso Regression modeli
    lasso_model = Lasso(alpha=alpha_)
    lasso_model.fit(X1_train, y1_train)
    y1_pred_lasso = lasso_model.predict(X1_test)
    mse_lasso = mean_squared_error(y1_test, y1_pred_lasso)

    # MSE değerlerini gösterme
    st.write(f"**Linear Regression MSE:** {mse_linear:.4f}")
    st.write(f"**Ridge Regression MSE (α={alpha_}):** {mse_ridge:.4f}")
    st.write(f"**Lasso Regression MSE (α={alpha_}):** {mse_lasso:.4f}")

    # Regresyon Eğrilerini Karşılaştırma
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X1_test, y1_test, color='black', label='Veri', alpha=0.6)
    ax.plot(X1_test, y1_pred_linear, color='blue', label='Linear Regression', linewidth=2)
    ax.plot(X1_test, y1_pred_ridge, color='red', label=f'Ridge Regression (α={alpha})', linewidth=2)
    ax.plot(X1_test, y1_pred_lasso, color='green', label=f'Lasso Regression (α={alpha})', linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Linear, Ridge ve Lasso Regresyon Eğrileri")
    ax.legend()

    st.pyplot(fig)

