#!/usr/bin/env python3
"""
Tests for Slide Manager Group Functionality

Verifies that the group endpoints work correctly with persistent storage:
- List groups
- Create group
- Get single group
- Update group
- Delete group
- Add slides to group
- Remove slide from group
- Bulk add slides to group

These tests use direct module loading to test the SlideMetadataManager
without requiring a running server.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_METADATA_PATH = REPO_ROOT / "src" / "enso_atlas" / "api" / "slide_metadata.py"

# Load the module directly
_spec = importlib.util.spec_from_file_location("slide_metadata", SLIDE_METADATA_PATH)
assert _spec and _spec.loader, "Failed to load slide_metadata module"
_slide_metadata = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _slide_metadata
_spec.loader.exec_module(_slide_metadata)

SlideMetadataManager = _slide_metadata.SlideMetadataManager
GroupResponse = _slide_metadata.GroupResponse


@pytest.fixture
def manager():
    """Create a SlideMetadataManager with a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    
    mgr = SlideMetadataManager(temp_path)
    yield mgr
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestGroupListAndCreate:
    """Test group listing and creation."""
    
    def test_list_groups_empty_initially(self, manager):
        """list_groups() should return empty list when no groups exist."""
        groups = manager.list_groups()
        assert groups == []
    
    def test_create_group_returns_group_response(self, manager):
        """create_group() should return a GroupResponse with correct fields."""
        group = manager.create_group("Test Group", "A test description")
        
        assert isinstance(group, GroupResponse)
        assert group.name == "Test Group"
        assert group.description == "A test description"
        assert group.id.startswith("grp_")
        assert group.slide_ids == []
        assert group.slide_count == 0
        assert group.created_at is not None
        assert group.updated_at is not None
    
    def test_create_group_persists_to_disk(self, manager):
        """Created groups should persist to disk."""
        manager.create_group("Persisted Group")
        
        # Create a new manager pointing to the same file
        manager2 = SlideMetadataManager(manager.storage_path)
        groups = manager2.list_groups()
        
        assert len(groups) == 1
        assert groups[0].name == "Persisted Group"
    
    def test_list_groups_returns_all_groups(self, manager):
        """list_groups() should return all created groups."""
        manager.create_group("Group 1")
        manager.create_group("Group 2")
        manager.create_group("Group 3")
        
        groups = manager.list_groups()
        
        assert len(groups) == 3
        names = {g.name for g in groups}
        assert names == {"Group 1", "Group 2", "Group 3"}


class TestGroupGetAndUpdate:
    """Test getting and updating individual groups."""
    
    def test_get_group_returns_group(self, manager):
        """get_group() should return the group with the given ID."""
        created = manager.create_group("My Group", "Description")
        
        fetched = manager.get_group(created.id)
        
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "My Group"
        assert fetched.description == "Description"
    
    def test_get_group_returns_none_for_invalid_id(self, manager):
        """get_group() should return None for non-existent ID."""
        result = manager.get_group("nonexistent-id")
        assert result is None
    
    def test_update_group_changes_name(self, manager):
        """update_group() should update the group name."""
        group = manager.create_group("Original Name")
        
        updated = manager.update_group(group.id, name="New Name")
        
        assert updated.name == "New Name"
        
        # Verify persistence
        fetched = manager.get_group(group.id)
        assert fetched.name == "New Name"
    
    def test_update_group_changes_description(self, manager):
        """update_group() should update the group description."""
        group = manager.create_group("Test", "Old description")
        
        updated = manager.update_group(group.id, description="New description")
        
        assert updated.description == "New description"
    
    def test_update_group_raises_for_invalid_id(self, manager):
        """update_group() should raise ValueError for non-existent ID."""
        with pytest.raises(ValueError, match="not found"):
            manager.update_group("nonexistent-id", name="New Name")


class TestGroupDelete:
    """Test group deletion."""
    
    def test_delete_group_removes_group(self, manager):
        """delete_group() should remove the group."""
        group = manager.create_group("To Delete")
        
        result = manager.delete_group(group.id)
        
        assert result is True
        assert manager.get_group(group.id) is None
    
    def test_delete_group_raises_for_invalid_id(self, manager):
        """delete_group() should raise ValueError for non-existent ID."""
        with pytest.raises(ValueError, match="not found"):
            manager.delete_group("nonexistent-id")
    
    def test_delete_group_removes_from_slides(self, manager):
        """delete_group() should remove group reference from all slides."""
        group = manager.create_group("Group to Delete")
        manager.add_slides_to_group(group.id, ["slide1", "slide2"])
        
        manager.delete_group(group.id)
        
        # Check that slides no longer reference the deleted group
        meta1 = manager.get_slide_metadata("slide1")
        meta2 = manager.get_slide_metadata("slide2")
        
        assert group.id not in meta1.groups
        assert group.id not in meta2.groups


class TestAddSlidesToGroup:
    """Test adding slides to groups."""
    
    def test_add_single_slide_to_group(self, manager):
        """add_slides_to_group() should add a single slide."""
        group = manager.create_group("Test Group")
        
        updated = manager.add_slides_to_group(group.id, ["slide1"])
        
        assert "slide1" in updated.slide_ids
        assert updated.slide_count == 1
    
    def test_add_multiple_slides_to_group(self, manager):
        """add_slides_to_group() should add multiple slides."""
        group = manager.create_group("Test Group")
        
        updated = manager.add_slides_to_group(group.id, ["slide1", "slide2", "slide3"])
        
        assert set(updated.slide_ids) == {"slide1", "slide2", "slide3"}
        assert updated.slide_count == 3
    
    def test_add_slides_updates_slide_metadata(self, manager):
        """add_slides_to_group() should update slide metadata with group reference."""
        group = manager.create_group("Test Group")
        manager.add_slides_to_group(group.id, ["slide1"])
        
        slide_meta = manager.get_slide_metadata("slide1")
        
        assert group.id in slide_meta.groups
    
    def test_add_slides_is_idempotent(self, manager):
        """Adding same slide twice should not duplicate it."""
        group = manager.create_group("Test Group")
        
        manager.add_slides_to_group(group.id, ["slide1"])
        updated = manager.add_slides_to_group(group.id, ["slide1"])
        
        assert updated.slide_ids.count("slide1") == 1
    
    def test_add_slides_raises_for_invalid_group(self, manager):
        """add_slides_to_group() should raise ValueError for non-existent group."""
        with pytest.raises(ValueError, match="not found"):
            manager.add_slides_to_group("nonexistent-id", ["slide1"])
    
    def test_add_slides_persists(self, manager):
        """Added slides should persist to disk."""
        group = manager.create_group("Persistent Group")
        manager.add_slides_to_group(group.id, ["slide1", "slide2"])
        
        # Create new manager to verify persistence
        manager2 = SlideMetadataManager(manager.storage_path)
        fetched = manager2.get_group(group.id)
        
        assert set(fetched.slide_ids) == {"slide1", "slide2"}


class TestRemoveSlideFromGroup:
    """Test removing slides from groups."""
    
    def test_remove_single_slide(self, manager):
        """remove_slide_from_group() should remove a slide."""
        group = manager.create_group("Test Group")
        manager.add_slides_to_group(group.id, ["slide1", "slide2"])
        
        updated = manager.remove_slide_from_group(group.id, "slide1")
        
        assert "slide1" not in updated.slide_ids
        assert "slide2" in updated.slide_ids
        assert updated.slide_count == 1
    
    def test_remove_slide_updates_slide_metadata(self, manager):
        """remove_slide_from_group() should update slide metadata."""
        group = manager.create_group("Test Group")
        manager.add_slides_to_group(group.id, ["slide1"])
        
        manager.remove_slide_from_group(group.id, "slide1")
        
        slide_meta = manager.get_slide_metadata("slide1")
        assert group.id not in slide_meta.groups
    
    def test_remove_slides_bulk(self, manager):
        """remove_slides_from_group() should remove multiple slides."""
        group = manager.create_group("Test Group")
        manager.add_slides_to_group(group.id, ["slide1", "slide2", "slide3"])
        
        updated = manager.remove_slides_from_group(group.id, ["slide1", "slide2"])
        
        assert updated.slide_ids == ["slide3"]
        assert updated.slide_count == 1
    
    def test_remove_slide_raises_for_invalid_group(self, manager):
        """remove_slide_from_group() should raise ValueError for non-existent group."""
        with pytest.raises(ValueError, match="not found"):
            manager.remove_slide_from_group("nonexistent-id", "slide1")


class TestBulkOperations:
    """Test bulk group operations."""
    
    def test_bulk_add_to_existing_group(self, manager):
        """Bulk add should work with existing group and multiple slides."""
        group = manager.create_group("Bulk Test")
        slides = ["slide1", "slide2", "slide3", "slide4", "slide5"]
        
        result = manager.add_slides_to_group(group.id, slides)
        
        assert len(result.slide_ids) == 5
        assert set(result.slide_ids) == set(slides)
    
    def test_bulk_add_incremental(self, manager):
        """Bulk add should work incrementally."""
        group = manager.create_group("Incremental")
        
        manager.add_slides_to_group(group.id, ["slide1", "slide2"])
        result = manager.add_slides_to_group(group.id, ["slide3", "slide4"])
        
        assert len(result.slide_ids) == 4


class TestResponseShapes:
    """Test that response shapes match frontend expectations."""
    
    def test_group_response_has_required_fields(self, manager):
        """GroupResponse should have all fields expected by frontend."""
        group = manager.create_group("Shape Test", "Description")
        
        # These fields are required by BackendGroup interface in frontend
        assert hasattr(group, 'id')
        assert hasattr(group, 'name')
        assert hasattr(group, 'description')
        assert hasattr(group, 'slide_ids')
        assert hasattr(group, 'created_at')
        assert hasattr(group, 'updated_at')
        
        # Verify types
        assert isinstance(group.id, str)
        assert isinstance(group.name, str)
        assert group.description is None or isinstance(group.description, str)
        assert isinstance(group.slide_ids, list)
        assert isinstance(group.created_at, str)
        assert isinstance(group.updated_at, str)
    
    def test_group_response_serializable(self, manager):
        """GroupResponse should be JSON serializable."""
        group = manager.create_group("Serializable", "Test")
        manager.add_slides_to_group(group.id, ["slide1"])
        
        fetched = manager.get_group(group.id)
        
        # Should be able to convert to dict and serialize
        data = fetched.dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        assert parsed["name"] == "Serializable"
        assert parsed["slide_ids"] == ["slide1"]


class TestEndpointCoverage:
    """Verify all expected API endpoints have backing functionality."""
    
    def test_list_groups_method_exists(self, manager):
        """GET /api/groups backing: list_groups()"""
        assert hasattr(manager, 'list_groups')
        assert callable(manager.list_groups)
    
    def test_create_group_method_exists(self, manager):
        """POST /api/groups backing: create_group()"""
        assert hasattr(manager, 'create_group')
        assert callable(manager.create_group)
    
    def test_get_group_method_exists(self, manager):
        """GET /api/groups/{id} backing: get_group()"""
        assert hasattr(manager, 'get_group')
        assert callable(manager.get_group)
    
    def test_update_group_method_exists(self, manager):
        """PATCH /api/groups/{id} backing: update_group()"""
        assert hasattr(manager, 'update_group')
        assert callable(manager.update_group)
    
    def test_delete_group_method_exists(self, manager):
        """DELETE /api/groups/{id} backing: delete_group()"""
        assert hasattr(manager, 'delete_group')
        assert callable(manager.delete_group)
    
    def test_add_slides_to_group_method_exists(self, manager):
        """POST /api/groups/{id}/slides backing: add_slides_to_group()"""
        assert hasattr(manager, 'add_slides_to_group')
        assert callable(manager.add_slides_to_group)
    
    def test_remove_slide_from_group_method_exists(self, manager):
        """DELETE /api/groups/{id}/slides/{slide_id} backing: remove_slide_from_group()"""
        assert hasattr(manager, 'remove_slide_from_group')
        assert callable(manager.remove_slide_from_group)
    
    def test_remove_slides_from_group_method_exists(self, manager):
        """Bulk remove backing: remove_slides_from_group()"""
        assert hasattr(manager, 'remove_slides_from_group')
        assert callable(manager.remove_slides_from_group)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
