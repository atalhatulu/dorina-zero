"""games/base_game.py — Oyun-Agnostik Soyut Arayüz (Abstract Base Game).
core/ motorunun tüm oyunlarla (Connect4, Othello, Dama vb.) konuşabilmesi için gereken sözleşme.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseGame(ABC):
    """Tüm oyunların uygulaması gereken standart arayüz.

    core/ motoru (MCTS, ReplayBuffer, Trainer) sadece bu sınıfı tanır;
    Connect4, Othello veya Dama'nın kurallarını bilmez.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Oyunun adı (örn: 'Connect4', 'Othello')."""
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Oyunun toplam olası hamle sayısı (örn: Connect4 için 7, Othello için 64)."""
        pass

    @property
    @abstractmethod
    def board_shape(self) -> tuple:
        """Sinir ağına girecek tensör boyutu (C, H, W) (örn: (3, 6, 7))."""
        pass

    @abstractmethod
    def get_initial_state(self):
        """Oyunun başlangıç tahtasını döndürür."""
        pass

    @abstractmethod
    def get_next_state(self, state, action: int, player: int):
        """Hamleyi uygular ve yeni tahtayı döndürür.

        Dönüş: yeni tahta nesnesi
        """
        pass

    @abstractmethod
    def get_valid_moves(self, state, player: int) -> np.ndarray:
        """O anki oyuncu için geçerli hamlelerin 1D boolean maskesi (boyut: action_space_size)."""
        pass

    @abstractmethod
    def check_terminal_value(self, state, player: int):
        """Oyun bitti mi kontrol eder.

        Dönüş:
            +1.0 : `player` kazandı
            -1.0 : `player` kaybetti (rakip kazandı)
             0.0 : Berabere
            None : Oyun henüz bitmedi
        """
        pass

    @abstractmethod
    def encode_state(self, state, player: int) -> np.ndarray:
        """Tahtayı sinir ağının beklediği (C, H, W) formatında float32 numpy dizisine çevirir."""
        pass

    @abstractmethod
    def clone_state(self, state):
        """MCTS simülasyonları için tahtanın bağımsız derin kopyasını döndürür."""
        pass

    @abstractmethod
    def get_canonical_state(self, state, player: int):
        """Tahtayı her zaman 'sıradaki oyuncu 1. oyuncudur' perspektifine çevirir."""
        pass

    @abstractmethod
    def state_to_key(self, state, player: int):
        """Transposition Table için benzersiz ve hash'lenebilir anahtar döndürür."""
        pass
