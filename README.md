import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import streamlit as st

class PacienteTerminal:
    def __init__(self, **kwargs):
        self.nombre = kwargs.get('nombre', '')
        self.edad = kwargs.get('edad', 0)
        self.genero = kwargs.get('genero', 'Otro')
        self.condicion = kwargs.get('condicion', '')
        self.dilema_etico = kwargs.get('dilema_etico', '')
        self.descripcion_caso = kwargs.get('descripcion_caso', '')
        self.tipo_dilema = kwargs.get('tipo_dilema', '')
        self.estado_funcional = kwargs.get('estado_funcional', 0)
        self.estado_cognitivo = kwargs.get('estado_cognitivo', 0)
        self.consciente = kwargs.get('consciente', False)
        self.directrices_anticipadas = kwargs.get('directrices_anticipadas', False)
        self.valores_paciente = kwargs.get('valores_paciente', False)
        self.preferencias_familia = kwargs.get('preferencias_familia', False)
        self.calidad_vida = kwargs.get('calidad_vida', 0)
        self.sufrimiento = kwargs.get('sufrimiento', 0)
        self.recursos = kwargs.get('recursos', 0)
        self.comunicacion_familia = kwargs.get('comunicacion_familia', 0)
        self.nivel_autonomia = kwargs.get('nivel_autonomia', 0)
        self.nivel_beneficencia = kwargs.get('nivel_beneficencia', 0)
        self.nivel_no_maleficencia = kwargs.get('nivel_no_maleficencia', 0)
        self.nivel_justicia = kwargs.get('nivel_justicia', 0)
        self.conflicto_justicia_no_maleficencia = kwargs.get('conflicto_justicia_no_maleficencia', False)
        self.conflicto_autonomia_beneficencia = kwargs.get('conflicto_autonomia_beneficencia', False)
        self.poblacion_vulnerable = kwargs.get('poblacion_vulnerable', False)
        self.cuidador_primario = kwargs.get('cuidador_primario', False)
        self.abandono_familiar = kwargs.get('abandono_familiar', False)
        self.junta_medica = kwargs.get('junta_medica', False)
        self.historia_clinica = kwargs.get('historia_clinica', '')
        self.fecha_diligenciamiento = kwargs.get('fecha_diligenciamiento', None)
        self.nombre_analista = kwargs.get('nombre_analista', '')
        self.ansiedades_miedos = kwargs.get('ansiedades_miedos', '')
        self.expectativas_esperanzas = kwargs.get('expectativas_esperanzas', '')
        self.autopercepcion_salud = kwargs.get('autopercepcion_salud', '')
        self.calidad_relaciones = kwargs.get('calidad_relaciones', '')
        self.conflictos_tensiones = kwargs.get('conflictos_tensiones', '')
        self.creencias_religiosas = kwargs.get('creencias_religiosas', '')
        self.sentido_proposito = kwargs.get('sentido_proposito', '')
        self.capacidad_entender = kwargs.get('capacidad_entender', '')
        self.deseos_preferencias = kwargs.get('deseos_preferencias', '')
        self.evolucion_caso = []
        self.bibliografia = []
        self.puntajes_principios = {
            "Respeto por la vida humana": 5,
            "Honestidad": 4,
            "Justicia": 4,
            "Respeto por los demás": 4,
            "Responsabilidad": 5,
            "Compasión": 5,
            "Libertad": 4,
            "Integridad": 4,
            "Tolerancia": 4,
            "Solidaridad": 4
        }

    def registrar_evolucion(self, fecha, estado_funcional, estado_cognitivo, calidad_vida, sufrimiento):
        self.evolucion_caso.append({
            "Fecha": fecha,
            "Estado Funcional": estado_funcional,
            "Estado Cognitivo": estado_cognitivo,
            "Calidad de Vida": calidad_vida,
            "Sufrimiento": sufrimiento
        })

    def mostrar_evolucion(self):
        if not self.evolucion_caso:
            return "No hay datos de evolución registrados."
        df_evolucion = pd.DataFrame(self.evolucion_caso)
        display(df_evolucion)
        return df_evolucion

    def agregar_bibliografia(self, referencia):
        self.bibliografia.append(referencia)
        print(f"Referencia bibliográfica '{referencia}' agregada.")

    def eliminar_bibliografia(self, referencia):
        if referencia in self.bibliografia:
            self.bibliografia.remove(referencia)
            print(f"Referencia bibliográfica '{referencia}' eliminada.")
        else:
            print(f"Referencia bibliográfica '{referencia}' no encontrada.")

    def mostrar_bibliografia(self):
        if not self.bibliografia:
            return "No hay referencias bibliográficas registradas."
        for ref in self.bibliografia:
            print(ref)

    def evaluacion_integral(self):
        if self.estado_funcional <= 2 and self.estado_cognitivo <= 2:
            return "Alta prioridad de cuidados paliativos."
        return "Evaluar más datos."

    def evaluar_autonomia(self):
        if not self.directrices_anticipadas:
            if self.valores_paciente or self.preferencias_familia:
                return "Consulta con la familia y respeto a los valores del paciente."
            return "Consultar más datos sobre los valores del paciente."
        return "Seguir directrices anticipadas del paciente."

    def evaluar_beneficencia_no_maleficencia(self):
        if self.calidad_vida <= 2 and self.sufrimiento >= 3:
            return "Evitar tratamientos que prolonguen el sufrimiento."
        return "Evaluar beneficios y riesgos de tratamientos avanzados."

    def evaluar_justicia(self):
        if self.recursos <= 2:
            return "Priorizar pacientes con mayores beneficios y probabilidades de recuperación."
        return "Evaluar distribución equitativa de recursos."

    def evaluar_comunicacion_decision_compartida(self):
        if self.comunicacion_familia >= 3:
            return "Fomentar la comunicación clara y empática con la familia."
        return "Mejorar estrategias de comunicación con la familia."

    def evaluar_rol_equipo_interdisciplinario(self):
        return "Promover el enfoque multidisciplinario y colaborativo."

    def evaluar_cuidados_paliativos(self):
        if self.calidad_vida <= 2:
            return "Integrar cuidados paliativos desde etapas tempranas."
        return "Evaluar necesidad de cuidados paliativos."

    def evaluar_objetivos_medicina(self):
        return {
            "Restablecimiento de la salud": 2,
            "Alivio de los síntomas": 5,
            "Restablecimiento de la función o conservación de la función alterada": 3,
            "Salvación de la vida en peligro": 2,
            "Mantenimiento del paciente mediante la orientación y la educación": 4
        }

    def recomendaciones_comite_etica(self):
        recomendaciones = []
        consideraciones_especiales = []

        if self.poblacion_vulnerable:
            consideraciones_especiales.append("Considerar la vulnerabilidad del paciente y asegurar un enfoque de cuidado especial.")
        if self.cuidador_primario:
            consideraciones_especiales.append("Involucrar al cuidador primario en las decisiones para asegurar una continuidad en el cuidado.")
        if self.abandono_familiar:
            consideraciones_especiales.append("Tomar medidas para proteger al paciente que ha sido abandonado por la familia, asegurando que sus derechos sean respetados.")

        if self.calidad_vida <= 2 and self.sufrimiento >= 3:
            recomendaciones.append("No iniciar diálisis, enfocar en cuidados paliativos.")
        else:
            recomendaciones.append("Evaluar beneficios y riesgos adicionales.")

        recomendaciones.append("Alinear la recomendación con el bienestar del paciente, la calidad de vida del paciente y la familia, los avances médicos y los objetivos de la medicina:")
        recomendaciones.extend([
            "1. Restablecimiento de la salud.",
            "2. Alivio de los síntomas.",
            "3. Restablecimiento de la función o conservación de la función alterada.",
            "4. Salvación de la vida en peligro.",
            "5. Mantenimiento del paciente mediante la orientación y la educación."
        ])

        if consideraciones_especiales:
            recomendaciones.append("\nConsideraciones Especiales:")
            recomendaciones.extend(consideraciones_especiales)

        return "\n".join(recomendaciones)

    def analizar_conflictos_bioetica(self):
        conflictos = []
        if self.conflicto_justicia_no_maleficencia:
            conflictos.append("Justicia Distributiva vs No Maleficencia")
        if self.conflicto_autonomia_beneficencia:
            conflictos.append("Autonomía vs Beneficencia")
        return conflictos

    def analizar_metodologias(self):
        anderson_diaz = {
            "Evaluación Integral del Paciente": "Evaluación integral y sistemática del paciente.",
            "Autonomía": "Consulta con la familia y respeto a los valores del paciente.",
            "Beneficencia y No Maleficencia": "Evaluación de la proporcionalidad de los tratamientos.",
            "Justicia": "Justicia en la distribución de recursos sanitarios.",
            "Comunicación y Decisión Compartida": "Comunicación clara y compartida.",
            "Rol del Equipo Interdisciplinario": "Perspectiva holística e interdisciplinaria.",
            "Cuidados Paliativos": "Enfoque en la calidad de vida y el alivio del sufrimiento."
        }
        diego_gracia = {
            "Evaluación Integral del Paciente": "Análisis detallado de la situación clínica del paciente.",
            "Autonomía": "Respeto por la autonomía y valores del paciente mediante la consulta con la familia.",
            "Beneficencia y No Maleficencia": "Evaluación rigurosa de los beneficios y riesgos de los tratamientos.",
            "Justicia": "Distribución equitativa de los recursos sanitarios."
        }
        return anderson_diaz, diego_gracia

    def evaluar_principios_morales_universales(self):
        return self.puntajes_principios

    def evaluar_tipo_recomendacion(self):
        if self.estado_funcional <= 2 and self.estado_cognitivo <= 2:
            return "Utilitarista", "Se prioriza la calidad de vida del paciente evitando tratamientos que prolonguen el sufrimiento."
        elif not self.directrices_anticipadas and (self.valores_paciente or self.preferencias_familia):
            return "Imperativo Categórico", "Se respeta la autonomía del paciente y las directrices familiares, siguiendo principios morales universales."
        else:
            return "Regla de Oro", "Se actúa considerando el trato que el paciente esperaría recibir, basándose en el respeto y la empatía."

    def evaluar_legalidad_colombia(self):
        derechos = []
        if self.estado_funcional <= 2 and self.estado_cognitivo <= 2:
            derechos.append("Derecho a cuidados paliativos (Ley 1733 de 2014, Art. 1)")
        if not self.directrices_anticipadas and (self.valores_paciente or self.preferencias_familia):
            derechos.append("Derecho a la autonomía y a la toma de decisiones informadas (Ley 1751 de 2015, Art. 6)")
        if self.calidad_vida <= 2 and self.sufrimiento >= 3:
            derechos.append("Derecho a morir con dignidad (Ley 1733 de 2014, Art. 2)")
        if self.tipo_dilema == 'Eutanasia':
            derechos.extend([
                "Sentencia C-239 de 1997: Despenalización de la eutanasia para pacientes en estado terminal.",
                "Resolución 1216 de 2015: Regulación del procedimiento para la eutanasia en Colombia."
            ])
        derechos.append("Constitución Política de Colombia de 1991: Artículos 11, 12 y 13, que garantizan el derecho a la vida, la integridad personal y la igualdad.")
        return derechos

    def evaluar_codigo_deontologico(self):
        articulos = []
        if self.estado_funcional <= 2 and self.estado_cognitivo <= 2:
            articulos.append("Artículo 18: El médico debe aplicar los cuidados paliativos adecuados para aliviar el sufrimiento del paciente.")
        if not self.directrices_anticipadas and (self.valores_paciente or self.preferencias_familia):
            articulos.append("Artículo 15: El médico debe respetar la autonomía del paciente en la toma de decisiones.")
        if self.calidad_vida <= 2 and self.sufrimiento >= 3:
            articulos.append("Artículo 34: El médico debe evitar la obstinación terapéutica.")
        return articulos

    def sanciones_por_no_seguir_recomendaciones(self):
        sanciones = {
            "Civiles": {
                "Descripción": "Posible demanda por negligencia médica y daños y perjuicios.",
                "Justificación": "Si no se siguen las recomendaciones y esto resulta en daño al paciente, la familia podría demandar por negligencia."
            },
            "Penales": {
                "Descripción": "Posible acusación por homicidio culposo o lesiones personales.",
                "Justificación": "Si la falta de acción o la acción incorrecta lleva a la muerte o daño severo del paciente, el médico podría enfrentar cargos penales."
            },
            "Administrativas": {
                "Descripción": "Multas y sanciones impuestas por autoridades sanitarias.",
                "Justificación": "Las autoridades sanitarias podrían imponer sanciones administrativas por no seguir los protocolos y estándares de cuidado."
            },
            "Deontológicas": {
                "Descripción": "Sanciones disciplinarias por parte del colegio médico, incluyendo suspensión o revocación de la licencia médica.",
                "Justificación": "El incumplimiento de las recomendaciones éticas y de los estándares profesionales podría llevar a sanciones por parte del colegio médico."
            }
        }

        # Determinar la sanción más pertinente y fuerte
        sancion_mas_pertinente = max(sanciones.values(), key=lambda x: len(x["Descripción"]))
        return sanciones, sancion_mas_pertinente

    def generar_reporte(self):
        conflictos = self.analizar_conflictos_bioetica()
        anderson_diaz, diego_gracia = self.analizar_metodologias()
        tipo_recomendacion, justificacion = self.evaluar_tipo_recomendacion()
        derechos_legales = self.evaluar_legalidad_colombia()
        articulos_deontologicos = self.evaluar_codigo_deontologico()
        sanciones, sancion_mas_pertinente = self.sanciones_por_no_seguir_recomendaciones()
        objetivos_medicina = self.evaluar_objetivos_medicina()
        principios_morales_universales = self.evaluar_principios_morales_universales()
        reporte = {
            "Fecha de Diligenciamiento": self.fecha_diligenciamiento,
            "Nombre del Analista": self.nombre_analista,
            "Número de Historia Clínica": self.historia_clinica,
            "Nombre": self.nombre,
            "Edad": self.edad,
            "Género": self.genero,
            "Condición": self.condicion,
            "Dilema Ético": self.dilema_etico,
            "Descripción del Caso": self.descripcion_caso,
            "Tipo de Dilema Perceptivo": self.tipo_dilema,
            "Estado del Paciente (Consciente/Inconsciente)": "Consciente" if self.consciente else "Inconsciente",
            "Población Vulnerable": "Sí" if self.poblacion_vulnerable else "No",
            "Cuidador Primario Permanente": "Sí" if self.cuidador_primario else "No",
            "Abandono Familiar": "Sí" if self.abandono_familiar else "No",
            "Junta Médica": "Sí" if self.junta_medica else "No",
            "Evaluación Integral del Paciente": self.evaluacion_integral(),
            "Autonomía": self.evaluar_autonomia(),
            "Beneficencia y No Maleficencia": self.evaluar_beneficencia_no_maleficencia(),
            "Justicia": self.evaluar_justicia(),
            "Comunicación y Decisión Compartida": self.evaluar_comunicacion_decision_compartida(),
            "Rol del Equipo Interdisciplinario": self.evaluar_rol_equipo_interdisciplinario(),
            "Cuidados Paliativos": self.evaluar_cuidados_paliativos(),
            "Recomendaciones del Comité de Ética": self.recomendaciones_comite_etica(),
            "Nivel de Autonomía": self.nivel_autonomia,
            "Nivel de Beneficencia": self.nivel_beneficencia,
            "Nivel de No Maleficencia": self.nivel_no_maleficencia,
            "Nivel de Justicia": self.nivel_justicia,
            "Conflictos entre Principios de Bioética": conflictos if conflictos else "No se encontraron conflictos entre los principios de bioética.",
            "Metodología de Anderson Díaz": anderson_diaz,
            "Metodología de Diego Gracia": diego_gracia,
            "Explicación de la Metodología de Anderson Díaz": "La metodología de Anderson Díaz resalta la importancia de una evaluación integral y sistemática del paciente, consultando con la familia y respetando los valores del paciente. Se enfatiza la comunicación clara y empática con la familia y la necesidad de un enfoque multidisciplinario y colaborativo. El objetivo es mejorar la calidad de vida del paciente y aliviar el sufrimiento, integrando cuidados paliativos desde etapas tempranas. En este caso, esto se aplica evaluando el estado funcional y cognitivo del paciente, consultando con la familia para respetar sus valores, y promoviendo un enfoque colaborativo en el tratamiento.",
            "Explicación de la Metodología de Diego Gracia": "La metodología de Diego Gracia subraya la importancia de evaluar la proporcionalidad de los tratamientos, respetando la autonomía del paciente y la consulta con la familia. Se debe priorizar la justicia en la distribución de recursos, asegurando que las decisiones se tomen de manera equitativa y basada en los principios éticos universales. En este caso, se aplica analizando detalladamente la situación clínica del paciente, evaluando los beneficios y riesgos de los tratamientos propuestos, y asegurando que se respeten los principios de autonomía y justicia en las decisiones médicas.",
            "Descripción de los Principios Morales Universales y su Grado de Cumplimiento": f"{principios_morales_universales}. Justificación: La evaluación integral del paciente permite identificar el grado de cumplimiento de los principios morales universales. Se prioriza el respeto por la vida humana y la compasión, al tiempo que se fomenta la honestidad y la integridad en la toma de decisiones.",
            "Tipo de Recomendación": tipo_recomendacion,
            "Justificación": justificacion,
            "Derechos Legales en Colombia": derechos_legales if derechos_legales else "No se encontraron alineaciones específicas con la legislación colombiana.",
            "Artículos del Código Deontológico": articulos_deontologicos if articulos_deontologicos else "No se encontraron alineaciones específicas con el Código Deontológico Médico.",
            "Elementos Antropológicos": "La recomendación se basa en el respeto a la dignidad humana, entendida como el valor intrínseco de cada persona, independientemente de su condición de salud. La dignidad humana exige un trato que respete la autonomía y valore la calidad de vida del paciente.",
            "Elementos Axiológicos": "Se valoran los principios morales y éticos universales, priorizando los valores del paciente y su familia, y respetando la integridad moral en la toma de decisiones.",
            "Elementos Teleológicos": "La finalidad de la recomendación es alcanzar el mejor resultado posible para el paciente, optimizando su calidad de vida y minimizando el sufrimiento, en armonía con los principios éticos y legales.",
            "Sanciones por no seguir las recomendaciones": sanciones,
            "Sanción Más Pertinente": sancion_mas_pertinente,
            "Objetivos de la Medicina": objetivos_medicina,
            "Paso a Paso de la Recomendación": "1. Evaluar el estado funcional y cognitivo del paciente.\n"
                                            "2. Consultar con la familia y respetar los valores del paciente.\n"
                                            "3. Evaluar beneficios y riesgos de tratamientos avanzados.\n"
                                            "4. Priorizar a los pacientes con mayores beneficios y probabilidades de recuperación.\n"
                                            "5. Fomentar la comunicación clara y empática con la familia.\n"
                                            "6. Promover el enfoque multidisciplinario y colaborativo.\n"
                                            "7. Integrar cuidados paliativos desde etapas tempranas.\n"
                                            "8. No iniciar diálisis y enfocar en cuidados paliativos si la calidad de vida es baja y el sufrimiento alto."
        }
        return reporte

    def generar_mapa_calor(self):
        data = pd.DataFrame.from_dict(self.puntajes_principios, orient='index', columns=['Puntaje'])
        sns.heatmap(data, annot=True, cmap='coolwarm')
        plt.title("Mapa de Calor: Correlación de Principios Morales Universales con Niveles Bioéticos")
        plt.show()

    def generar_graficas(self):
        # Gráfica Niveles de Principios Bioéticos
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame.from_dict(self.puntajes_principios, orient='index', columns=['Puntaje'])
        sns.barplot(x=data.index, y='Puntaje', data=data, palette='viridis')
        plt.title("Niveles de Principios Bioéticos")
        plt.xlabel("Principios")
        plt.ylabel("Niveles")
        plt.xticks(rotation=45)
        plt.show()

        # Gráfica Grado de Cumplimiento de los Principios Morales Universales
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame.from_dict(self.puntajes_principios, orient='index', columns=['Puntaje'])
        sns.lineplot(data=data, marker='o', dashes=False)
        plt.title("Grado de Cumplimiento de los Principios Morales Universales")
        plt.xlabel("Principios")
        plt.ylabel("Grado de Cumplimiento")
        plt.xticks(rotation=45)
        plt.show()

        # Gráfica Niveles de Cumplimiento de los Objetivos de la Medicina
        plt.figure(figsize=(10, 6))
        objetivos = self.evaluar_objetivos_medicina()
        data = pd.DataFrame.from_dict(objetivos, orient='index', columns=['Nivel'])
        sns.barplot(x=data.index, y='Nivel', data=data, palette='magma')
        plt.title("Niveles de Cumplimiento de los Objetivos de la Medicina")
        plt.xlabel("Objetivos")
        plt.ylabel("Niveles")
        plt.xticks(rotation=45)
        plt.show()

# Código principal para ejecutar el modelo

def main():
    st.title("Modelo para la Toma de Decisiones Clínicas y Bioéticas para Pacientes Terminales")
    st.write("Autor: Dr. Anderson Díaz Pérez")

    st.header("Datos del Paciente")
    nombre = st.text_input("Nombre")
    edad = st.slider("Edad", 0, 120, 0)
    genero = st.selectbox("Género", ['Masculino', 'Femenino', 'Otro'])
    condicion = st.text_area("Condición")
    dilema_etico = st.text_area("Dilema Ético")
    descripcion_caso = st.text_area("Descripción del Caso")
    tipo_dilema = st.selectbox("Tipo de Dilema", ['Cuidado paliativo', 'Obstinación terapéutica', 'Limitación del esfuerzo terapéutico', 'Sedación paliativa', 'Muerte y duelo en la familia', 'Conflictos de intereses entre el equipo médico y la familia del paciente inconsciente', 'Evitar el dolor y sufrimiento', 'Eutanasia', 'Dilemas relacionados con el trasplante de órganos'])
    estado_funcional = st.slider("Estado Funcional", 0, 5, 0)
    estado_cognitivo = st.slider("Estado Cognitivo", 0, 5, 0)
    consciente = st.checkbox("Consciente")
    directrices_anticipadas = st.checkbox("Directrices Anticipadas")
    valores_paciente = st.checkbox("Valores del Paciente")
    preferencias_familia = st.checkbox("Preferencias Familiares")
    calidad_vida = st.slider("Calidad de Vida", 0, 5, 0)
    sufrimiento = st.slider("Sufrimiento", 0, 5, 0)
    recursos = st.slider("Recursos", 0, 5, 0)
    comunicacion_familia = st.slider("Comunicación Familiar", 0, 5, 0)
    nivel_autonomia = st.slider("Nivel Autonomía", 0, 5, 0)
    nivel_beneficencia = st.slider("Nivel Beneficencia", 0, 5, 0)
    nivel_no_maleficencia = st.slider("Nivel No Maleficencia", 0, 5, 0)
    nivel_justicia = st.slider("Nivel Justicia", 0, 5, 0)
    conflicto_justicia_no_maleficencia = st.checkbox("Justicia vs No Maleficencia")
    conflicto_autonomia_beneficencia = st.checkbox("Autonomía vs Beneficencia")
    poblacion_vulnerable = st.checkbox("Población Vulnerable")
    cuidador_primario = st.checkbox("Cuidador Primario Permanente")
    abandono_familiar = st.checkbox("Abandono Familiar")
    junta_medica = st.checkbox("Junta Médica")
    historia_clinica = st.text_input("Historia Clínica")
    fecha_diligenciamiento = st.date_input("Fecha Diligenciamiento")
    nombre_analista = st.text_input("Nombre del Analista")
    ansiedades_miedos = st.text_area("Ansiedades y Miedos")
    expectativas_esperanzas = st.text_area("Expectativas y Esperanzas")
    autopercepcion_salud = st.text_area("Autopercepción de Salud")
    calidad_relaciones = st.text_area("Calidad de las Relaciones")
    conflictos_tensiones = st.text_area("Conflictos y Tensiones")
    creencias_religiosas = st.text_area("Creencias Religiosas")
    sentido_proposito = st.text_area("Sentido de Propósito")
    capacidad_entender = st.text_area("Capacidad de Entender y Decidir")
    deseos_preferencias = st.text_area("Deseos y Preferencias")
    registrar_evolucion = st.checkbox("Registrar Evolución")
    fecha_evolucion = st.date_input("Fecha Evolución")
    estado_funcional_evolucion = st.slider("Estado Funcional (Evolución)", 0, 5, 0)
    estado_cognitivo_evolucion = st.slider("Estado Cognitivo (Evolución)", 0, 5, 0)
    calidad_vida_evolucion = st.slider("Calidad de Vida (Evolución)", 0, 5, 0)
    sufrimiento_evolucion = st.slider("Sufrimiento (Evolución)", 0, 5, 0)
    bibliografia = st.text_area("Referencias Bibliográficas")

    if st.button("Generar Reporte"):
        datos_paciente = {
            "nombre": nombre,
            "edad": edad,
            "genero": genero,
            "condicion": condicion,
            "dilema_etico": dilema_etico,
            "descripcion_caso": descripcion_caso,
            "tipo_dilema": tipo_dilema,
            "estado_funcional": estado_funcional,
            "estado_cognitivo": estado_cognitivo,
            "consciente": consciente,
            "directrices_anticipadas": directrices_anticipadas,
            "valores_paciente": valores_paciente,
            "preferencias_familia": preferencias_familia,
            "calidad_vida": calidad_vida,
            "sufrimiento": sufrimiento,
            "recursos": recursos,
            "comunicacion_familia": comunicacion_familia,
            "nivel_autonomia": nivel_autonomia,
            "nivel_beneficencia": nivel_beneficencia,
            "nivel_no_maleficencia": nivel_no_maleficencia,
            "nivel_justicia": nivel_justicia,
            "conflicto_justicia_no_maleficencia": conflicto_justicia_no_maleficencia,
            "conflicto_autonomia_beneficencia": conflicto_autonomia_beneficencia,
            "poblacion_vulnerable": poblacion_vulnerable,
            "cuidador_primario": cuidador_primario,
            "abandono_familiar": abandono_familiar,
            "junta_medica": junta_medica,
            "historia_clinica": historia_clinica,
            "fecha_diligenciamiento": fecha_diligenciamiento,
            "nombre_analista": nombre_analista,
            "ansiedades_miedos": ansiedades_miedos,
            "expectativas_esperanzas": expectativas_esperanzas,
            "autopercepcion_salud": autopercepcion_salud,
            "calidad_relaciones": calidad_relaciones,
            "conflictos_tensiones": conflictos_tensiones,
            "creencias_religiosas": creencias_religiosas,
            "sentido_proposito": sentido_proposito,
            "capacidad_entender": capacidad_entender,
            "deseos_preferencias": deseos_preferencias,
            "bibliografia": bibliografia
        }

        paciente = PacienteTerminal(**datos_paciente)

        if registrar_evolucion:
            paciente.registrar_evolucion(fecha_evolucion, estado_funcional_evolucion, estado_cognitivo_evolucion, calidad_vida_evolucion, sufrimiento_evolucion)
            df_evolucion = paciente.mostrar_evolucion()
            df_evolucion.to_excel('evolucion_paciente_terminal.xlsx', index=False)
            st.success("Evolución registrada y guardada como 'evolucion_paciente_terminal.xlsx'")
        else:
            reporte = paciente.generar_reporte()
            df = pd.DataFrame(list(reporte.items()), columns=['Dimensión Evaluada', 'Recomendación'])
            st.write(df)
            df.to_excel('reporte_paciente_terminal.xlsx', index=False)
            st.success("Reporte generado y guardado como 'reporte_paciente_terminal.xlsx'")
            paciente.generar_mapa_calor()
            paciente.generar_graficas()

        # Análisis de sentimientos
        sentimiento = analizar_sentimientos(descripcion_caso)
        st.write("Análisis de Sentimientos:", sentimiento)

        # Análisis de frecuencia de palabras
        frecuencia_palabras = analizar_frecuencia_palabras(descripcion_caso)
        st.write("Análisis de Frecuencia de Palabras:", frecuencia_palabras)

        # Clustering
        clustering = analizar_clustering(descripcion_caso)
        st.write("Clustering:", clustering)

        # Resumen automático
        resumen = resumen_automatico(descripcion_caso)
        st.write("Resumen Automático:", resumen)

def analizar_sentimientos(texto):
    blob = TextBlob(texto)
    sentimiento = blob.sentiment.polarity
    if sentimiento > 0:
        return "Positivo"
    elif sentimiento < 0:
        return "Negativo"
    else:
        return "Neutro"

def analizar_frecuencia_palabras(texto):
    stop_words = list(set(stopwords.words('spanish')))
    vectorizer = CountVectorizer(stop_words=stop_words)
    try:
        X = vectorizer.fit_transform([texto])
        palabra_frecuencia = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
        palabra_frecuencia_ordenada = dict(sorted(palabra_frecuencia.items(), key=lambda item: item[1], reverse=True))
        return palabra_frecuencia_ordenada
    except ValueError:
        return "El texto proporcionado solo contiene palabras vacías."

def analizar_clustering(texto):
    stop_words = list(set(stopwords.words('spanish')))
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform([texto])
    if X.shape[0] >= 3:  # Asegurarse de que haya suficientes muestras para realizar clustering
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        return kmeans.labels_
    else:
        return []

def resumen_automatico(texto):
    parser = PlaintextParser.from_string(texto, Tokenizer("spanish"))
    summarizer = LsaSummarizer()
    resumen = summarizer(parser.document, 3)  # Resumir en 3 oraciones
    return " ".join([str(sentence) for sentence in resumen])

if __name__ == "__main__":
    main()
