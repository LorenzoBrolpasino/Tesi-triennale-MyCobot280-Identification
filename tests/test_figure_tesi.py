"""Test della generazione delle figure aggiuntive per l'elaborato."""

import pandas as pd
import pytest

from scripts.genera_figure_tesi import coppie_amax_per_verso


# Costruisce le due righe di una coppia andata/ritorno per un giunto.
def _coppia(sessione, giunto, amax_pos, amax_neg, id_pos=True, id_neg=True):
    comune = {"session": sessione, "joint_name": giunto, "commanded_velocity": 100,
              "success": True}
    return [
        {**comune, "delta_q": 4.2, "amax_fit": amax_pos, "amax_identifiable": id_pos},
        {**comune, "delta_q": -4.2, "amax_fit": amax_neg, "amax_identifiable": id_neg},
    ]


class TestCoppieAmax:
    def test_tiene_le_coppie_identificabili(self):
        # Entrambi i versi identificabili: la coppia entra nel confronto.
        fit = pd.DataFrame(_coppia("1_100", "j1", 8.0, 8.4))
        coppie, scartate = coppie_amax_per_verso(fit)

        assert len(coppie) == 1
        assert scartate == 0
        assert coppie.iloc[0].amax_positivo == 8.0
        assert coppie.iloc[0].amax_negativo == 8.4
        assert coppie.iloc[0].asimmetria_perc == pytest.approx(100 * 0.4 / 8.0)

    def test_scarta_se_un_verso_non_e_identificabile(self):
        # Basta un verso non identificabile per buttare via l'intera coppia.
        fit = pd.DataFrame(_coppia("1_100", "j1", 8.0, 90.0, id_neg=False))
        coppie, scartate = coppie_amax_per_verso(fit)

        assert len(coppie) == 0
        assert scartate == 1

    def test_un_solo_movimento_non_fa_coppia(self):
        # Un giunto con un solo verso non e' una coppia: non si conta fra le scartate.
        fit = pd.DataFrame(_coppia("1_100", "j1", 8.0, 8.4)[:1])
        coppie, scartate = coppie_amax_per_verso(fit)

        assert len(coppie) == 0
        assert scartate == 0

    def test_conta_separatamente_giunti_e_sessioni(self):
        # Coppie diverse per giunto e per sessione restano distinte.
        righe = (_coppia("1_100", "j1", 8.0, 8.1)
                 + _coppia("1_100", "j2", 7.9, 8.2)
                 + _coppia("2_100", "j1", 8.3, 7.7, id_pos=False))
        coppie, scartate = coppie_amax_per_verso(pd.DataFrame(righe))

        assert len(coppie) == 2
        assert scartate == 1
        assert set(coppie.session) == {"1_100"}

    def test_sceglie_la_colonna_indicata(self):
        # Con a_acc e a_dec diversi, il parametro colonna deve decidere quale si legge.
        righe = _coppia("1_100", "j1", 8.0, 8.4)
        for r, acc, dec in zip(righe, (8.0, 8.4), (7.0, 7.6)):
            r["a_acc_fit"], r["a_dec_fit"] = acc, dec
        fit = pd.DataFrame(righe)

        acc, _ = coppie_amax_per_verso(fit, colonna="a_acc_fit")
        dec, _ = coppie_amax_per_verso(fit, colonna="a_dec_fit")

        assert (acc.iloc[0].amax_positivo, acc.iloc[0].amax_negativo) == (8.0, 8.4)
        assert (dec.iloc[0].amax_positivo, dec.iloc[0].amax_negativo) == (7.0, 7.6)
        # Il default resta amax_fit, per non cambiare il comportamento del trapezoidale.
        predefinito, _ = coppie_amax_per_verso(fit)
        assert predefinito.iloc[0].amax_positivo == 8.0

    def test_ignora_i_fit_non_convergenti(self):
        # I movimenti con success=False non partecipano all'accoppiamento.
        righe = _coppia("1_100", "j1", 8.0, 8.1)
        righe[1]["success"] = False
        coppie, scartate = coppie_amax_per_verso(pd.DataFrame(righe))

        assert len(coppie) == 0
        assert scartate == 0
