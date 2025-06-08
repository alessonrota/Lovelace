| Pasta / arquivo                                   | Tamanho                                                                                                                               | Função                                                                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **`ada-lovelace-lora\adapter_model.safetensors`** | **≈ 80 MB**                                                                                                                           | **Pesos da sua LoRA** – os deltas que o treinamento gerou em relação ao Vicuna-7B. É isso que “transforma” o modelo base. |
| `ada-lovelace-lora\adapter_config.json`           | alguns bytes                                                                                                                          | Metadados da LoRA (r, α, target modules etc.). O loader precisa desse arquivo para saber como injetar os pesos.           |
| `ada-lovelace-lora\tokenizer.*`                   | 4 MB                                                                                                                                  | Cópia do tokenizer do Vicuna – incluída automaticamente pelo `Trainer`.                                                   |
| `ada-lovelace-lora\checkpoint-3\ …`               | \~80 MB                                                                                                                               | *Checkpoint* extra salvo no fim do treino; pode ser apagado se não quiser versões intermediárias.                         |
| **`hf_dataset\`**                                 | (vários arquivos Arrow + JSON)                                                                                                        | Dataset Hugging Face gerado pelo `build_dataset.py`.                                                         |
| **`offload\`**                                    | sub-pasta criada quando você rodou o script com `device_map="auto"`; armazena temporariamente partes do modelo que não cabem na VRAM. |                                                                                                                           |
| **scripts (`train_lora*.py`, `teste*.py`)**       | –                                                                                                                                     | Seus pipelines de treino e teste.                                                                                         |
| **`phi2-lora\`**                                  | outra LoRA que você começou (base Phi-2).                                                                                             |                                                                                                                           |

## Books_Corpus

BERNARDES, Ieda Pimenta; RIBEIRO, Camila Giovana; PEREIRA, Maria Elisa (org.).  
**Planos de classificação e tabelas de temporalidade de documentos dos órgãos da administração pública direta do Estado de São Paulo: atividades-fim (2007–2022)**.  
1. reimpressão da 2. ed. São Paulo: Arquivo Público do Estado de São Paulo, 2024. 448 p.  
ISBN 978-85-6935-314-0.

**Plano de classificação e tabelas de temporalidade de documentos para as administrações públicas municipais: atividades meio e fim**.  
1. ed. São Paulo: Arquivo Público do Estado de São Paulo, 2023. 206 p.  
ISBN 978-85-6935-3.

**Política de gestão e preservação de documentos digitais**.  
São Paulo: Arquivo Público do Estado de São Paulo, 2022. 129 p.  
ISBN 978-85-6935-312-6.

**Política pública de arquivos e gestão documental do Estado de São Paulo**.  
4. ed., rev. e ampl. São Paulo: Arquivo Público do Estado de São Paulo, 2022. 416 p.  
ISBN 978-85-6935-311-9.

**Guia para a avaliação dinâmica da massa documental acumulada: eliminação rápida e segura de documentos de arquivo**.  
1. ed. São Paulo: Arquivo Público do Estado de São Paulo, 2021. 97 p.  
ISBN 978-85-6935-309-6.

**Cartilha: implantação da gestão documental nos municípios**.  
1. ed. São Paulo: Arquivo Público do Estado de São Paulo, 2023. 78 p.  
(adaptação de *Guia técnico de transparência municipal*, Arquivo Público do Estado e TCE-SP, 2018).

**Política pública de arquivos e gestão documental do Estado de São Paulo**.  
3. ed., rev. e ampl. São Paulo: Arquivo Público do Estado de São Paulo, 2022. 404 p.  
ISBN 978-85-6935-310-2.

**Guia técnico de transparência municipal**.  
São Paulo: Arquivo Público do Estado de São Paulo; Departamento de Gestão do SAESP; Departamento de Difusão e Preservação do Acervo; Centro de Assistência aos Municípios; Tribunal de Contas do Estado de São Paulo; Diretoria Geral; Departamentos de Supervisão e Fiscalização, 2019. 176 p.  
ISBN 978-85-6935-306-5.

**Serviços de informações ao cidadão – SIC: primeiros passos**.  
1. ed. São Paulo: Arquivo Público do Estado de São Paulo, 2019. 72 p.  
ISBN 978-85-6935-304-1.

**Plano de classificação e tabela de temporalidade de documentos da administração pública do Estado de São Paulo: atividades meio**.  
2. ed., rev. amp., 1. reimpressão. São Paulo: Arquivo Público do Estado de São Paulo, 2019. 264 p.  
ISBN 978-85-6935-302-7.

**MODELO DE PLANO DE CLASSIFICAÇÃO E TABELA DE TEMPORALIDADE DE DOCUMENTOS DO PODER LEGISLATIVO MUNICIPAL (ATIVIDADES-FIM E MEIO)**.  
São Paulo: Arquivo Público do Estado de São Paulo; Câmara Municipal de Barueri, 2018. 388 p.  
ISBN 978-85-6935-305-8.

**ROTEIRO PARA ELABORAÇÃO DE PLANO DE CLASSIFICAÇÃO E TABELA DE TEMPORALIDADE DE DOCUMENTOS DAS ATIVIDADES-FIM**.  
Coleção Gestão Documental, nº 2. São Paulo: Departamento de Gestão do SAESP, 2018. 44 p.  
ISBN 978-85-6935-303-4.

**PLANO DE CLASSIFICAÇÃO E TABELA DE TEMPORALIDADE DA ADMINISTRAÇÃO PÚBLICA DO ESTADO DE SÃO PAULO: ATIVIDADES-MEIO**.  
2. ed. rev. e ampl. São Paulo: Arquivo Público do Estado de São Paulo e colaboradores, 2018. 264 p.  
ISBN 978-85-6935-302-7.

**MODELO DE PLANO DE CLASSIFICAÇÃO E TABELA DE TEMPORALIDADE DE DOCUMENTOS DO PODER LEGISLATIVO MUNICIPAL (ATIVIDADES-FIM)**.  
São Paulo: Departamento de Gestão do SAESP; Sistema de Arquivos do Estado de São Paulo, 2015. 70 p.  
ISBN 978-85-6935-301-0.

**PROCEDIMENTOS PARA RECOLHIMENTO DE DOCUMENTOS DE GUARDA PERMANENTE**.  
Coleção Gestão Documental, nº 1. São Paulo: Arquivo Público do Estado de São Paulo, 2014. 53 p.  
ISBN 978-85-6935-300-3.

**ARQUIVO EM IMAGENS**.  
Série Última Hora - Popular/Populismo. São Paulo: Arquivo Público do Estado de São Paulo, 2014. 123 p.  
ISBN 978-85-6344-315-1.

**MEMÓRIA HISTÓRICA DA CAPITANIA DE SÃO PAULO**.  
Edição e estudo de Renata Ferreira Costa. São Paulo: Arquivo Público do Estado de São Paulo, 2014. 177 p.  
ISBN 978-85-6344-313-7.

**POLÍTICA PÚBLICA DE ARQUIVOS E GESTÃO DOCUMENTAL DO ESTADO DE SÃO PAULO**.  
2. ed. São Paulo: Arquivo Público do Estado de São Paulo, 2014. 248 p.  
ISBN 978-85-6344-309-0.

**MANUAL DE NORMAS E PROCEDIMENTOS DE PROTOCOLO PARA A ADMINISTRAÇÃO PÚBLICA DO ESTADO DE SÃO PAULO**.  
São Paulo: Arquivo Público do Estado de São Paulo, 2013. 116 p.  
ISBN 978-85-6344-308-3.

**HISTÓRIAS DA (I)MIGRAÇÃO: IMIGRANTES E MIGRANTES EM SÃO PAULO...**  
Coleção Ensino & Memória. São Paulo: Arquivo Público do Estado de São Paulo, 2013. 252 p.  
ISBN 978-85-6344-307-6.

**ANAIS BRASILEIROS E BRASILIANISTAS: NOVAS GERAÇÕES, NOVOS OLHARES – UMA HOMENAGEM A EMILIA VIOTTI DA COSTA**.  
São Paulo: Arquivo Público do Estado de São Paulo (Org.), 2012. 154 p.  
ISBN 978-85-6344-312-0.

**TEMPOS DE FASCISMOS: IDEOLOGIA – INTOLERÂNCIA – IMAGINÁRIO**.  
1. ed. São Paulo: Edusp; Imprensa Oficial; Arquivo do Estado de São Paulo, 2010. 504 p.  
ISBN 978-85-6344-302-1.

**HISTÓRIAS DO FUTEBOL**.  
Coleção Ensino & Memória. São Paulo: Arquivo Público do Estado de São Paulo, 2010. 192 p.  
ISBN 978-85-6344-301-4.

**OS CORTIÇOS DE SANTA IFIGÊNIA: SANITARISMO E URBANIZAÇÃO (1893)**.  
São Paulo: Arquivo Público do Estado de São Paulo; Imprensa Oficial, 2010. 244 p.  
ISBN 978-85-6344-300-7.

**VIAGEM PELA CARTOGRAFIA DO TERRITÓRIO PAULISTA**.  
São Paulo: Instituto Geográfico e Cartográfico do Estado de São Paulo; Secretaria de Economia e Planejamento, 2010. 189 p.  
ISBN 978-85-6203-105-2.

**HISTÓRIA DO ESTADO DE SÃO PAULO / A FORMAÇÃO DA UNIDADE PAULISTA – GOVERNO E MUNICIPALIDADE**.  
São Paulo: Arquivo Público do Estado de São Paulo; Editora UNESP, 2010. 302 p.  
ISBN 978-85-6203-104-5.

**HISTÓRIA DO ESTADO DE SÃO PAULO / A FORMAÇÃO DA UNIDADE PAULISTA – REPÚBLICA**.  
São Paulo: Arquivo Público do Estado de São Paulo; Editora UNESP, 2010. 678 p.  
ISBN 978-85-6203-103-8.

**HISTÓRIA DO ESTADO DE SÃO PAULO / A FORMAÇÃO DA UNIDADE PAULISTA – COLÔNIA E IMPÉRIO**.  
São Paulo: Arquivo Público do Estado de São Paulo; Editora UNESP, 2010. 504 p.  
ISBN 978-85-6203-102-1.

**A LUTA PELA ANISTIA**.  
São Paulo: Arquivo Público do Estado de São Paulo, 2009. 488 p.  
ISBN 978-85-6203-101-4.

**PRESERVAÇÃO DE ACERVOS BIBLIOGRÁFICOS**.  
Homenagem a Guita Mindlin. São Paulo: Arquivo Público do Estado de São Paulo, 2008. 84 p.  
ISBN 978-85-7060-681-5.
