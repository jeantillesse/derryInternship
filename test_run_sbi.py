import unittest
import torch

# Import de la fonction à tester
from run_sbi import simulateur_sbi_hybride

class TestRunSBI(unittest.TestCase):
    def test_simulateur_sbi_hybride(self):
        # Création d'un batch de 2 jeux de paramètres
        # [N_ampa, N_nmda, N_ca*, L_neck]
        theta = torch.tensor([
            [120.0, 15.0, 3.0, 0.2],
            [100.0, 10.0, 5.0, 1.0]
        ])
        
        # Exécution du simulateur
        results = simulateur_sbi_hybride(theta)
        
        # Vérifications
        self.assertTrue(isinstance(results, torch.Tensor), "Le résultat doit être un torch.Tensor")
        self.assertEqual(results.shape, (2, 14), "La forme attendue est (batch_size=2, num_outputs=14)")
        self.assertTrue(torch.all(torch.isfinite(results)), "Toutes les valeurs retournées doivent être finies")

if __name__ == '__main__':
    unittest.main()
