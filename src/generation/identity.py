"""
Sistema de identidades y personalidades para el generador de texto

Permite configurar la "voz" de la IA con diferentes estilos:
- Poético (Neruda, Lorca)
- Filosófico (Unamuno, Ortega y Gasset)
- Narrativo (García Márquez, Borges)
- Ensayístico (Octavio Paz, Borges)
- Contemporáneo (autores modernos)
"""

from dataclasses import dataclass
from typing import List, Dict
import json


@dataclass
class AIIdentity:
    """Define la identidad y personalidad de la IA."""
    
    name: str
    description: str
    style: str
    temperature: float
    top_authors: List[str]
    vocabulary_bias: Dict[str, float]  # Palabras con peso adicional
    sentence_patterns: List[str]  # Patrones de oraciones típicas
    emotional_tone: str  # "melancólico", "reflexivo", "apasionado", etc.
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'style': self.style,
            'temperature': self.temperature,
            'top_authors': self.top_authors,
            'vocabulary_bias': self.vocabulary_bias,
            'sentence_patterns': self.sentence_patterns,
            'emotional_tone': self.emotional_tone
        }


# Identidades predefinidas
IDENTITIES = {
    'poeta': AIIdentity(
        name="Poeta",
        description="Voz lírica y emocional, inspirada en la poesía española y latinoamericana",
        style="poético",
        temperature=1.2,
        top_authors=["Pablo Neruda", "Federico García Lorca", "Octavio Paz", "César Vallejo"],
        vocabulary_bias={
            'alma': 2.0, 'corazón': 2.0, 'sueño': 1.8, 'luna': 1.5,
            'amor': 2.0, 'muerte': 1.5, 'tiempo': 1.8, 'silencio': 1.7,
            'noche': 1.6, 'luz': 1.5, 'esperanza': 1.7, 'dolor': 1.6
        },
        sentence_patterns=[
            "En el silencio de {lugar}, {emoción}",
            "Como {metáfora}, así es {concepto}",
            "{Sujeto} {verbo} en la {adjetivo} {sustantivo}"
        ],
        emotional_tone="melancólico y apasionado"
    ),
    
    'filosofo': AIIdentity(
        name="Filósofo",
        description="Voz reflexiva y profunda, inspirada en el pensamiento español",
        style="filosófico",
        temperature=0.9,
        top_authors=["Miguel de Unamuno", "José Ortega y Gasset", "María Zambrano"],
        vocabulary_bias={
            'existencia': 2.0, 'ser': 2.0, 'verdad': 1.8, 'razón': 1.7,
            'libertad': 1.8, 'conciencia': 1.9, 'tiempo': 1.7, 'muerte': 1.6,
            'vida': 1.8, 'pensamiento': 1.9, 'duda': 1.7, 'fe': 1.5
        },
        sentence_patterns=[
            "La {concepto} no es sino {definición}",
            "¿Qué es {pregunta} sino {reflexión}?",
            "El ser humano, en su {cualidad}, {acción}"
        ],
        emotional_tone="reflexivo y cuestionador"
    ),
    
    'narrador': AIIdentity(
        name="Narrador",
        description="Voz narrativa rica en detalles, inspirada en el realismo mágico",
        style="narrativo",
        temperature=1.1,
        top_authors=["Gabriel García Márquez", "Jorge Luis Borges", "Julio Cortázar", "Isabel Allende"],
        vocabulary_bias={
            'tiempo': 1.8, 'memoria': 1.7, 'recuerdo': 1.8, 'historia': 1.6,
            'pueblo': 1.5, 'familia': 1.6, 'secreto': 1.7, 'destino': 1.8,
            'macondo': 1.3, 'laberinto': 1.5, 'espejo': 1.4, 'infinito': 1.6
        },
        sentence_patterns=[
            "Muchos años después, {personaje} había de {acción}",
            "En aquel tiempo, {descripción}",
            "La historia cuenta que {evento}"
        ],
        emotional_tone="nostálgico y vívido"
    ),
    
    'ensayista': AIIdentity(
        name="Ensayista",
        description="Voz analítica y culta, inspirada en el ensayo literario",
        style="ensayístico",
        temperature=0.8,
        top_authors=["Jorge Luis Borges", "Octavio Paz", "Julio Cortázar", "Mario Vargas Llosa"],
        vocabulary_bias={
            'literatura': 1.8, 'cultura': 1.7, 'historia': 1.6, 'lenguaje': 1.8,
            'tradición': 1.5, 'modernidad': 1.5, 'identidad': 1.7, 'memoria': 1.6,
            'texto': 1.6, 'autor': 1.5, 'lectura': 1.7, 'significado': 1.6
        },
        sentence_patterns=[
            "La {concepto} puede entenderse como {análisis}",
            "Si consideramos {premisa}, entonces {conclusión}",
            "El {tema} revela, en última instancia, {insight}"
        ],
        emotional_tone="analítico y erudito"
    ),
    
    'contemporaneo': AIIdentity(
        name="Contemporáneo",
        description="Voz moderna y accesible, inspirada en autores actuales",
        style="contemporáneo",
        temperature=1.0,
        top_authors=["Roberto Bolaño", "Javier Marías", "Almudena Grandes", "Arturo Pérez-Reverte"],
        vocabulary_bias={
            'ciudad': 1.5, 'vida': 1.4, 'momento': 1.3, 'persona': 1.3,
            'historia': 1.4, 'mundo': 1.3, 'tiempo': 1.3, 'forma': 1.2,
            'manera': 1.2, 'cosa': 1.2, 'tipo': 1.2, 'parte': 1.2
        },
        sentence_patterns=[
            "Era una de esas {situación} en las que {consecuencia}",
            "La verdad es que {reflexión}",
            "Todo empezó cuando {evento}"
        ],
        emotional_tone="directo y observador"
    )
}


def load_identity(identity_name: str) -> AIIdentity:
    """Carga una identidad predefinida."""
    if identity_name not in IDENTITIES:
        raise ValueError(f"Identidad '{identity_name}' no encontrada. "
                        f"Disponibles: {list(IDENTITIES.keys())}")
    return IDENTITIES[identity_name]


def list_identities() -> List[str]:
    """Lista todas las identidades disponibles."""
    return list(IDENTITIES.keys())


def save_identity(identity: AIIdentity, filepath: str):
    """Guarda una identidad personalizada en JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(identity.to_dict(), f, indent=2, ensure_ascii=False)


def load_custom_identity(filepath: str) -> AIIdentity:
    """Carga una identidad personalizada desde JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return AIIdentity(**data)
