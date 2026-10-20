from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import json
import os
import matplotlib.pyplot as plt
from reportlab.lib.utils import ImageReader
import io

def gerar_grafico_sentado_este_pe(timeline):
    timeline = sorted(timeline, key=lambda x: x['timestamp'])
    tempos = [item['timestamp'] for item in timeline]
    estados = [0 if item['estado'] == 'sentado' else 1 for item in timeline]

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Sentado', 'De Pé'])
    ax.set_xlabel('Tempo (s)')
    ax.set_title('Timeline Sentado / De Pé')
    ax.set_facecolor('white')  


    ax.plot(tempos, estados, 'ko', zorder=3)

    # segmentos coloridos entre os pontos
    for i in range(len(tempos)-1):
        x_vals = [tempos[i], tempos[i+1]]
        y_vals = [estados[i], estados[i+1]]
        color = 'green' if estados[i] == 1 else 'red'
        ax.plot(x_vals, y_vals, color=color, linewidth=3, zorder=2)

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def gerar_grafico_inclinacao_tronco(angles, limiar, tempos):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title('Inclinação do Tronco ao Longo do Tempo')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Ângulo (°)')
    ax.set_ylim(0, max(angles) + 10)

    # Cor por segmento: verde se <= limiar, vermelho se > limiar
    for i in range(len(tempos) - 1):
        x = tempos[i:i+2]
        y = angles[i:i+2]
        cor = 'red' if max(y) > limiar else 'green'
        ax.plot(x, y, color=cor, linewidth=3)

    # Pontos para visualização
    ax.plot(tempos, angles, 'ko')
    ax.axhline(y=limiar, color='gray', linestyle='--', label=f'Limiar {limiar}°')
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def gerar_grafico_posicao_bracos(tempos, ombro_dir, cotovelo_dir, ombro_esq, cotovelo_esq):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title('Posição dos Braços (altura cotovelo vs ombro)')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Coordenada vertical (y)')

    ax.plot(tempos, ombro_dir, label='Ombro Direito', color='blue', linestyle='--')
    ax.plot(tempos, cotovelo_dir, label='Cotovelo Direito', color='blue', linestyle='-')

    ax.plot(tempos, ombro_esq, label='Ombro Esquerdo', color='orange', linestyle='--')
    ax.plot(tempos, cotovelo_esq, label='Cotovelo Esquerdo', color='orange', linestyle='-')

    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def escreve_secao_avaliacao(c, y, width, height, margem_esq, titulo, texto, img_buf):
    """
    Adiciona uma secção no PDF com título, texto e imagem (gráfico).
    Retorna a nova posição y.
    """
    espacamento = 15
    def escreve_linha(texto, tamanho=12, negrito=False, quebra_linha=True):
        nonlocal y
        if y - espacamento < 2 * cm:
            c.showPage()
            y = height - 2 * cm
        fonte = "Helvetica-Bold" if negrito else "Helvetica"
        c.setFont(fonte, tamanho)
        c.drawString(margem_esq, y, texto)
        if quebra_linha:
            y -= espacamento

    escreve_linha(titulo, tamanho=13, negrito=True)
    for linha in texto.split('\n'):
        escreve_linha(linha, tamanho=11)

    img = ImageReader(img_buf)
    img_width = 16 * cm
    img_height = 4 * cm
    if y - img_height < 2 * cm:
        c.showPage()
        y = height - 2 * cm
    c.drawImage(img, margem_esq, y - img_height, width=img_width, height=img_height)
    y -= (img_height + espacamento)

    return y



def gerar_pdf_do_json(json_path, output_dir="relatorios_pdf"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    pdf_path = os.path.join(output_dir, f"{data['session_id']}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    margem_esq = 2 * cm
    y = height - 2 * cm

    def escreve_linha(texto, tamanho=12, negrito=False, quebra_linha=True, espacamento=15):
        nonlocal y
        # Verifica espaço para nova linha
        if y - espacamento < 2 * cm:
            c.showPage()
            y = height - 2 * cm
        fonte = "Helvetica-Bold" if negrito else "Helvetica"
        c.setFont(fonte, tamanho)
        c.drawString(margem_esq, y, texto)
        if quebra_linha:
            y -= espacamento

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 3 * cm, "Relatório de Análise de Ergonomia")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, height - 4 * cm, f"ID da Sessão: {data['session_id']}")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, height - 5 * cm, f"Duração Total: {data['duration_sec']} segundos")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, height - 6 * cm, f"Número de Pessoas Monitoradas: {len(data.get('sitting_stats', {}))}")

    c.showPage()  
    y = height - 2 * cm
    
    # Para cada pessoa, mostra estatísticas + gráfico juntos
    for person_id in data.get("sitting_stats", {}).keys():
        stats = data["sitting_stats"][person_id]
        timeline = data.get("timeline", {}).get(person_id, [])
        angulos_data = data.get("angles_tronco", {}).get(person_id, [])
        posicao_bracos_data = data.get("posicao_bracos", {}).get(person_id, [])

        escreve_linha(f"ID {person_id}:", tamanho=13, negrito=True)
        escreve_linha(f"     → Sentado: {stats['sitting_percent']}%")
        escreve_linha(f"     → De Pé: {stats['standing_percent']}%")
        escreve_linha(f"     → Amostras: {stats['samples']}")
        escreve_linha("")

        if len(timeline) > 1:
            img_buf = gerar_grafico_sentado_este_pe(timeline)
            img = ImageReader(img_buf)
            img_width = 16 * cm
            img_height = 4 * cm
            if y - img_height < 2 * cm:
                c.showPage()
                y = height - 2 * cm
            c.drawImage(img, margem_esq, y - img_height, width=img_width, height=img_height)
            y -= (img_height + 20)
        else:
            escreve_linha("    Timeline insuficiente para gráfico")
        escreve_linha("")

        if angulos_data:
            tempos = [item['timestamp'] for item in angulos_data]
            angles = [item['angle'] for item in angulos_data]
            limiar = data.get("posture_threshold", 15)  
            img_buf = gerar_grafico_inclinacao_tronco(angles, limiar, tempos)
            y = escreve_secao_avaliacao(c, y, width, height, margem_esq,
                                        "Inclinação do Tronco",
                                        "Análise da flexão lateral do tronco ao longo do tempo.",
                                        img_buf)

        
        if posicao_bracos_data:
            tempos = [item['timestamp'] for item in posicao_bracos_data]
            ombro_dir = [item['ombro_dir'] for item in posicao_bracos_data]
            cotovelo_dir = [item['cotovelo_dir'] for item in posicao_bracos_data]
            ombro_esq = [item['ombro_esq'] for item in posicao_bracos_data]
            cotovelo_esq = [item['cotovelo_esq'] for item in posicao_bracos_data]

            img_buf = gerar_grafico_posicao_bracos(tempos, ombro_dir, cotovelo_dir, ombro_esq, cotovelo_esq)
            y = escreve_secao_avaliacao(c, y, width, height, margem_esq,
                                        "Posição dos Braços",
                                        "Altura relativa dos ombros e cotovelos ao longo do tempo.",
                                        img_buf)


    c.save()
    return pdf_path
