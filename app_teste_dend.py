import streamlit as st
from streamlit import rerun
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial import ConvexHull, distance
from matplotlib.patches import Polygon
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Configuração inicial do estado da sessão
if 'page' not in st.session_state:
    st.session_state.page = "questionario"
if 'run_fast' not in st.session_state:
    st.session_state.run_fast = False
if 'run_slow' not in st.session_state:
    st.session_state.run_slow = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Arquivo para salvar as respostas
arquivo_csv = "respostas_hogwarts.csv"

def questionario():
    st.title("🧙‍♂️ Questionário de Perfil Hogwarts")
    
    st.markdown("""
### Responda às afirmações abaixo em uma escala de **1 a 5**, onde:
- **1 = Discordo totalmente**
- **3 = Neutro**
- **5 = Concordo totalmente**
""")
    st.markdown("---")
    # Pergunta 1
    st.markdown("### 1️⃣ Em situações difíceis, eu costumo enfrentar os desafios, mesmo que sinta medo.")
    q1 = st.slider(" ", 1, 5, 3)
    
    # Pergunta 2
    st.markdown("### 2️⃣ Costumo aprender coisas novas com facilidade.")
    q2 = st.slider("  ", 1, 5, 3)
    
    # Pergunta 3
    st.markdown("### 3️⃣ Eu costumo valorizar muito relações de confiança.")
    q3 = st.slider("   ", 1, 5, 3)
    
    # Pergunta 4
    st.markdown("### 4️⃣ Gosto de pensar em ideias novas ou maneiras diferentes de fazer as coisas.")
    q4 = st.slider("    ", 1, 5, 3)
    st.markdown("---")

    if st.button("Enviar respostas"):
        st.session_state.user_data = {
            "Bravery": q1,
            "Intelligence": q2,
            "Loyalty": q3,
            "Creativity": q4
        }
        st.session_state.submitted = True
        st.session_state.page = "home_page"
        st.success("✅ Respostas enviadas com sucesso!")
        st.rerun()

# Carregar e preparar os dados
def load_data():
    df_original = pd.read_csv('hogwarts_characters.csv', 
                              usecols=['Bravery', 'Intelligence', 'Loyalty', 'Creativity'])
    
    if 'user_data' in st.session_state:
        df_user = pd.DataFrame([{
            'Bravery': st.session_state.user_data['Bravery'],
            'Intelligence': st.session_state.user_data['Intelligence'],
            'Loyalty': st.session_state.user_data['Loyalty'],
            'Creativity': st.session_state.user_data['Creativity']
        }])
        df_combined = pd.concat([df_original, df_user], ignore_index=True)
    else:
        df_combined = df_original
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_combined.values)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    Z = linkage(distance.pdist(X), method='complete')

    return X, X_2d, Z, df_combined.columns, df_combined.values, len(df_original)
X, X_2d, Z, feature_names, original_values, n_original = load_data()


# Reduzir para 2D usando PCA para visualização
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Matriz de ligação hierárquica (usando os dados originais 4D)
Z = linkage(distance.pdist(X), method='complete')

def determine_house(cluster_values):
    """Determina a casa com base nas características médias do cluster"""
    avg_bravery = cluster_values[:, 0].mean()
    avg_intelligence = cluster_values[:, 1].mean()
    avg_loyalty = cluster_values[:, 2].mean()
    avg_creativity = cluster_values[:, 3].mean()
    
    # Calcula scores para cada casa
    scores = {
        "Grifinória": avg_bravery * 3 + avg_loyalty * 1.5,
        "Sonserina": avg_intelligence * 2 + avg_creativity * 1.5 + avg_bravery * 0.5,
        "Corvinal": avg_intelligence * 2.5 + avg_creativity * 2,
        "Lufa-Lufa": avg_loyalty * 3 + avg_bravery * 1.5
    }
    
    # Encontra a casa com maior score
    return max(scores.items(), key=lambda x: x[1])[0]

def get_house_color(house):
    """Retorna a cor correspondente à casa"""
    colors = {
        "Grifinória": "#AE0001",  # Vermelho
        "Sonserina": "#2A623D",   # Verde
        "Corvinal": "#222F5B",    # Azul
        "Lufa-Lufa": "#FFDB00"    # Amarelo
    }
    return colors.get(house, "#000000")

def plot_clusters_and_dendrogram(n_clusters, house_mapping=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Gráfico de clusters - usando as componentes principais para visualização 2D
    clusters = cut_tree(Z, n_clusters=n_clusters).flatten()
    ax1.clear()
    ax1.set_title(f"Número de grupos: {n_clusters}", fontsize=14, pad=20)
    ax1.set_xlabel("Componente Principal 1", fontsize=12)
    ax1.set_ylabel("Componente Principal 2", fontsize=12)
    ax1.grid(ls='--', alpha=0.3)
    
    # Cores personalizadas se houver mapeamento de casas
    if house_mapping:
        colors = [get_house_color(house_mapping.get(c, "")) for c in clusters[:n_original]]
        user_colors = [get_house_color(house_mapping.get(c, "")) for c in clusters[n_original:]]
        cmap = None
    else:
        colors = clusters[:n_original]
        user_colors = clusters[n_original:]
        cmap = 'tab20'
    
    # Plotar pontos originais
    scatter = ax1.scatter(X_2d[:n_original, 0], X_2d[:n_original, 1], 
                         c=colors, cmap=cmap, s=20, alpha=0.7, label='Personagens')
    
    # Plotar APENAS O ÚLTIMO ponto do usuário
    # Plotar ponto do usuário (apenas o último)
    if n_original < len(X_2d):
        # Garante que estamos pegando sempre o mesmo ponto do usuário
        user_point = X_2d[-1]
        user_cluster = clusters[-1]
        
        user_color = (get_house_color(house_mapping[user_cluster]) 
                     if house_mapping else plt.cm.tab20(user_cluster % 20))
        
        ax1.scatter(user_point[0], user_point[1], 
                   c=[user_color], s=150, alpha=1.0,
                   edgecolors='black', linewidths=2,
                   marker='*', label='Você')
        ax1.legend()

        ax1.annotate("Você", 
             xy=user_point,
             xytext=(10, 10),
             textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))
    
    # Convex Hull para cada cluster (com tratamento de erro)
    for cluster_id in np.unique(clusters):
        cluster_points = X_2d[clusters == cluster_id]
        if len(cluster_points) >= 3:  # ConvexHull precisa de pelo menos 3 pontos
            try:
                hull = ConvexHull(cluster_points, qhull_options='QJ')  # QJ para jitter
                color = (get_house_color(house_mapping[cluster_id]) 
                         if house_mapping else plt.cm.tab20(cluster_id % 20))
                
                hull_polygon = Polygon(
                    cluster_points[hull.vertices],
                    closed=True,
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.4,
                    lw=1.5,
                    linestyle = '--'
                )
                ax1.add_patch(hull_polygon)
                
                # Adicionar label da casa
                if house_mapping:
                    centroid = cluster_points.mean(axis=0)
                    ax1.text(centroid[0], centroid[1], house_mapping[cluster_id],
                             fontsize=10, weight='bold', ha='center', va='center',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            except:
                pass
    
    # Dendrograma
    ax2.clear()
    if n_clusters > 1:
        cut_height = Z[-n_clusters+1, 2]
    else:
        cut_height = 0
    
    dendrogram(
        Z,
        ax=ax2,
        color_threshold=cut_height,
        above_threshold_color='black',
        leaf_font_size=8,
        orientation='top'
    )
    
    if n_clusters > 1:
        ax2.axhline(y=cut_height, color='r', linestyle='--', linewidth=1.5, 
                   label=f'Corte para {n_clusters} grupos')
        ax2.legend(fontsize=10)
    
    ax2.set_title('Dendrograma', fontsize=14, pad=20)
    ax2.set_xlabel('Índice do Estudante', fontsize=12)
    ax2.set_ylabel('Distância', fontsize=12)
    
    plt.tight_layout()
    return fig

house_icons = {
    "Grifinória": "🦁",
    "Sonserina": "🐍", 
    "Corvinal": "🦅",
    "Lufa-Lufa": "🦡"
}
def get_house_icon(house):
    return house_icons.get(house, "🏰")

def get_house_description(house):
    descriptions = {
        "Grifinória": "Onde habitam os corajosos! Valorizamos bravura e determinação.",
        "Sonserina": "A casa dos astutos e ambiciosos. Bem-vindo aos espertinhos!",
        "Corvinal": "Para os sábios e curiosos. O conhecimento é nosso maior tesouro.",
        "Lufa-Lufa": "O lar dos leais e trabalhadores. Aqui valorizamos a dedicação."
    }
    return descriptions.get(house, "Bem-vindo à sua nova casa em Hogwarts!")

def home_page():
    st.title("Agrupamento de Estudantes de Hogwarts")
    st.write("""
    Esta aplicação agrupa os estudantes de Hogwarts com base em suas características:
    Coragem, Inteligência, Lealdade e Criatividade.
    """)
    
    # Mostrar 4 clusters inicialmente (um para cada casa)
    n_clusters = 4
    clusters = cut_tree(Z, n_clusters=n_clusters).flatten()
    
    # Determinar a casa para cada cluster
    house_mapping = {}
    house_counts = {"Grifinória": 0, "Sonserina": 0, "Corvinal": 0, "Lufa-Lufa": 0}
    
    # Primeiro passada: calcular scores para todos os clusters
    cluster_scores = []
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_values = original_values[cluster_indices]
        scores = {
            "Grifinória": cluster_values[:, 0].mean() * 3 + cluster_values[:, 2].mean() * 1.5,
            "Sonserina": cluster_values[:, 1].mean() * 2 + cluster_values[:, 3].mean() * 1.5,
            "Corvinal": cluster_values[:, 1].mean() * 2.5 + cluster_values[:, 3].mean() * 2,
            "Lufa-Lufa": cluster_values[:, 2].mean() * 3 + cluster_values[:, 0].mean() * 1.5
        }
        cluster_scores.append((cluster_id, scores))
    
    # Ordenar clusters pelo score total (para atribuir casas mais distintas primeiro)
    cluster_scores.sort(key=lambda x: sum(x[1].values()), reverse=True)
    
    # Atribuir casas garantindo que todas sejam representadas
    available_houses = ["Grifinória", "Sonserina", "Corvinal", "Lufa-Lufa"]
    for cluster_id, scores in cluster_scores:
        # Se ainda temos casas não atribuídas, forçar uma nova casa
        if available_houses:
            # Encontrar a casa disponível com maior score para este cluster
            best_house = max(available_houses, key=lambda h: scores[h])
            house_mapping[cluster_id] = best_house
            available_houses.remove(best_house)
        else:
            # Todas as casas já foram atribuídas pelo menos uma vez
            house_mapping[cluster_id] = max(scores.items(), key=lambda x: x[1])[0]
    
    fig = plot_clusters_and_dendrogram(n_clusters, house_mapping)
    st.pyplot(fig)
    
    st.balloons()  # Isso vai lançar confetes!
    # Determinar a casa do usuário (último ponto)
    user_house = None
    if n_original < len(clusters):
        user_cluster = clusters[-1]
        user_house = house_mapping.get(user_cluster, "Desconhecida")
        
        # Mensagem de parabéns estilizada com ícone
        st.markdown(f"""
        <div style="
            background-color: {get_house_color(user_house)};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h2 style="color: white; margin: 0;">🏰 Parabéns! 🏰</h2>
            <h1 style="color: white; margin: 10px 0;">Você agora faz parte da</h1>
            <h1 style="color: white; margin: 10px 0; font-size: 2.5em;">
                {get_house_icon(user_house)} {user_house.upper()} {get_house_icon(user_house)}
            </h1>
            <p style="color: white; margin: 0;">✨ {get_house_description(user_house)} ✨</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()  # Efeito de confetes
    
    if st.button("Animação que ilustra o processo de agrupamento"):
        st.session_state.page = "fast_animation"
        rerun()

def fast_animation_page():
    st.title("Agrupamento Hierárquico")
    placeholder = st.empty()
    
    if not st.session_state.run_fast:
        st.session_state.run_fast = True
        for n_clusters in range(50, 9, -4):
            with placeholder.container():
                fig = plot_clusters_and_dendrogram(n_clusters)
                st.pyplot(fig)
            time.sleep(0.001)
    
    if st.button("Continuar animação"):
        st.session_state.page = "slow_animation"
        st.session_state.run_fast = True
        rerun()

def slow_animation_page():
    st.title("Agrupamento Hierárquico")
    placeholder = st.empty()
    
    if not st.session_state.run_slow:
        st.session_state.run_slow = True
        for n_clusters in range(10, 0, -1):
            with placeholder.container():
                fig = plot_clusters_and_dendrogram(n_clusters)
                st.pyplot(fig)
            time.sleep(0.5)
    
    if st.button("Voltar ao Início"):
        st.session_state.page = "home_page"
        st.session_state.run_slow = False
        rerun()

# Controle de páginas
if st.session_state.page == "questionario":
    questionario()
elif st.session_state.page == "home_page":
    # Limpar o cache para garantir que os novos dados sejam carregados
    st.cache_data.clear()
    home_page()
elif st.session_state.page == "fast_animation":
    fast_animation_page()
elif st.session_state.page == "slow_animation":
    slow_animation_page()